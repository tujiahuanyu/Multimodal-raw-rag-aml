from dataclasses import replace
import json

import numpy as np
import pytest

from evaluation.src.aml.associations import refresh_links
from evaluation.src.aml.contract_smoke import validate_search
from evaluation.src.aml.providers import ServiceError
from evaluation.src.aml.store import Store, digest
from test_aml_api import (
    add_payload, backend, client, image_part, search_payload, settings,
)


def insert_chunks(store, entries, count=5):
    with store.connection() as connection:
        connection.execute('BEGIN IMMEDIATE')
        for user, identifier, vector in entries:
            vector = np.asarray(vector, dtype='<f4')
            connection.execute(
                'INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)',
                (identifier, user, identifier, 0, 'Synthetic ' + identifier,
                 json.dumps(['message-' + identifier]), vector.tobytes(), len(vector)),
            )
        for user in {entry[0] for entry in entries}:
            refresh_links(connection, user, {entry[1] for entry in entries if entry[0] == user}, count)
        connection.commit()


def links(store, user='user-1'):
    with store.connection() as connection:
        return [tuple(row) for row in connection.execute(
            'SELECT source_id,target_id,similarity FROM chunk_links WHERE user_id=? '
            'ORDER BY source_id,similarity DESC,target_id DESC', (user,),
        )]


def test_incremental_links_equal_exact_nearest_neighbors(settings):
    store = Store(settings.data_dir, settings.identity())
    try:
        vectors = np.random.default_rng(31).normal(size=(17, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for start, end in [(0, 4), (4, 9), (9, 17)]:
            insert_chunks(store, [('user-1', f'chunk-{number:02}', vectors[number]) for number in range(start, end)])
        expected = []
        for number, vector in enumerate(vectors):
            ranked = sorted(
                ((float(other @ vector), f'chunk-{position:02}') for position, other in enumerate(vectors) if position != number),
                reverse=True,
            )[:5]
            expected.extend((f'chunk-{number:02}', target, score) for score, target in ranked)
        actual = links(store)
        assert [(source, target) for source, target, score in actual] == [(source, target) for source, target, score in expected]
        np.testing.assert_allclose([row[2] for row in actual], [row[2] for row in expected], atol=1e-6)
        assert all(source != target for source, target, score in actual)
    finally:
        store.close()


def test_new_chunk_updates_old_links_and_scopes_are_isolated(settings):
    store = Store(settings.data_dir, settings.identity())
    try:
        insert_chunks(store, [('user-1', 'first', [1, 0]), ('user-1', 'distant', [0, 1]), ('other', 'foreign', [1, 0])], count=1)
        assert links(store) == [('distant', 'first', 0.0), ('first', 'distant', 0.0)]
        insert_chunks(store, [('user-1', 'nearby', [1, .01])], count=1)
        assert next(target for source, target, score in links(store) if source == 'first') == 'nearby'
        assert not links(store, 'other')
        assert all('foreign' not in row[:2] for row in links(store))
    finally:
        store.close()


def test_image_views_share_one_graph_node_with_maximum_similarity(settings):
    store = Store(settings.data_dir, settings.identity())
    try:
        insert_chunks(store, [('user-1', 'red', [1, 0]), ('user-1', 'blue', [0, 1])], count=1)
        image_node = digest(('image', 'user-1', 'image-message', 'pixels'))
        with store.connection() as connection:
            for number, vector in enumerate([[1, 0], [0, 1]]):
                connection.execute('INSERT INTO image_views VALUES (?,?,?,?,?,?,?,?,?,?)', (
                    str(number), 'user-1', 'image-session', 'image-message', 'pixels',
                    'generated', number, 'Synthetic image view', np.asarray(vector, dtype='<f4').tobytes(), 2,
                ))
            refresh_links(connection, 'user-1', {image_node}, 1)
        assert next(target for source, target, score in links(store) if source == 'red') == image_node
        assert next(target for source, target, score in links(store) if source == 'blue') == image_node
        assert len({source for source, target, score in links(store)}) == 3
    finally:
        store.close()


def test_links_persist_and_purge_preserves_other_users(settings):
    store = Store(settings.data_dir, settings.identity())
    insert_chunks(store, [('user-1', 'first', [1, 0]), ('user-1', 'second', [1, .1]), ('other', 'third', [0, 1]), ('other', 'fourth', [.1, 1])])
    before = links(store)
    other = links(store, 'other')
    store.close()
    store = Store(settings.data_dir, settings.identity())
    try:
        assert links(store) == before
        store.purge_user('user-1')
        assert links(store) == []
        assert links(store, 'other') == other
    finally:
        store.close()


def test_rechunking_removes_stale_links(client):
    assert client.post('/v1/memories/add', json=add_payload(content='red camera in Paris ' * 300)).status_code == 200
    store = client.app.state.engine.store
    with store.connection() as connection:
        old = {row[0] for row in connection.execute('SELECT chunk_id FROM chunks')}
    assert client.post('/v1/memories/add', json=add_payload(request='append', content='blue ' * 300)).status_code == 200
    with store.connection() as connection:
        current = {row[0] for row in connection.execute('SELECT chunk_id FROM chunks')}
    assert old - current
    assert all(source in current and target in current for source, target, score in links(store))
    assert len(links(store)) == len(current) * min(5, len(current) - 1)


def test_association_failure_rolls_back_add_atomically(client, monkeypatch):
    assert client.post('/v1/memories/add', json=add_payload(content='red ' * 300)).status_code == 200
    store = client.app.state.engine.store
    before = links(store)
    messages = store.messages('user-1', 'session-1')

    def fail(connection, *arguments):
        connection.execute('DELETE FROM chunk_links')
        raise ServiceError(502, 'association_test_failure')

    monkeypatch.setattr('evaluation.src.aml.associations.refresh_links', fail)
    assert client.post('/v1/memories/add', json=add_payload(request='append', content='blue ' * 100)).status_code == 502
    assert links(store) == before
    assert store.messages('user-1', 'session-1') == messages


@pytest.mark.parametrize('top_k', [0, 1, 15, 44, 100])
def test_fifteen_anchors_expand_two_each_then_deduplicate(client, backend, monkeypatch, top_k):
    payload = add_payload()
    payload['messages'] = [{'role': 'user', 'content': f'Original source {number}.'} for number in range(45)]
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    store = client.app.state.engine.store
    messages = store.messages('user-1', 'session-1')
    candidates = [{'chunk_id': f'node-{number}', 'content': message['render'], 'source_ids': [message['message_id']]} for number, message in enumerate(messages)]
    with store.connection() as connection:
        connection.execute('DELETE FROM chunk_links WHERE user_id=?', ('user-1',))
        for number in range(15):
            for offset in range(2):
                target = 15 + 2 * number + offset
                connection.execute('INSERT INTO chunk_links VALUES (?,?,?,?,?)', (
                    'user-1', f'node-{number}', f'node-{target}', .9 - offset * .1,
                    json.dumps([messages[target]['message_id']]),
                ))
    monkeypatch.setattr(store, 'candidates', lambda *arguments: candidates[:15])
    response = client.post('/v1/memories/search', json={**search_payload(), 'top_k': top_k})
    assert response.status_code == 200
    data = response.json()
    validate_search(data, top_k)
    assert len(data['data']) == min(top_k, 45)
    assert len(backend.rerank_inputs) == int(top_k > 0)
    if top_k > 0:
        assert backend.rerank_limits == [15]
    assert all('Original source' in row['content'] for row in data['data'])
    if top_k == 100:
        assert {row['id'] for row in data['data']} == {row['message_id'] for row in messages}
        assert [row['id'] for row in data['data'][:15]] == [row['message_id'] for row in messages[:15]]


def test_overlapping_neighbors_are_deduplicated_and_not_recursively_expanded(client, backend, monkeypatch):
    payload = add_payload()
    payload['messages'] = [{'role': 'user', 'content': f'Original {number}.'} for number in range(5)]
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    store = client.app.state.engine.store
    messages = store.messages('user-1', 'session-1')
    candidates = [{'chunk_id': f'node-{number}', 'content': message['render'], 'source_ids': [message['message_id']]} for number, message in enumerate(messages[:2])]
    with store.connection() as connection:
        connection.execute('DELETE FROM chunk_links WHERE user_id=?', ('user-1',))
        for source, target in [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 4)]:
            source_ids = [messages[2 if target == 3 else target]['message_id']]
            connection.execute('INSERT INTO chunk_links VALUES (?,?,?,?,?)', ('user-1', f'node-{source}', f'node-{target}', .1 if (source, target) == (0, 4) else .9, json.dumps(source_ids)))
    monkeypatch.setattr(store, 'candidates', lambda *arguments: candidates)
    response = client.post('/v1/memories/search', json=search_payload())
    assert response.status_code == 200
    assert {row['id'] for row in response.json()['data']} == {message['message_id'] for message in messages[:3]}
    assert len(backend.rerank_inputs) == 1


def test_image_anchor_expansion_keeps_original_pixels_and_other_session(client, backend, monkeypatch):
    payload = add_payload(content=[{'type': 'text', 'text': 'Original caption.'}, image_part()])
    assert client.post('/v1/memories/add', json=payload).status_code == 200
    assert client.post('/v1/memories/add', json=add_payload(request='second', session='other-session', content='Original related fact.')).status_code == 200
    store = client.app.state.engine.store
    with store.connection() as connection:
        image = connection.execute('SELECT message_id,image_hash FROM image_views LIMIT 1').fetchone()
    identifier = digest(('image', 'user-1', image['message_id'], image['image_hash']))
    other = store.messages('user-1', 'other-session')[0]
    with store.connection() as connection:
        connection.execute('DELETE FROM chunk_links WHERE user_id=?', ('user-1',))
        connection.execute('INSERT INTO chunk_links VALUES (?,?,?,?,?)', ('user-1', identifier, 'other-node', .9, json.dumps([other['message_id']])))
    monkeypatch.setattr(store, 'candidates', lambda *arguments: [{'chunk_id': identifier, 'content': 'Generated retrieval view.', 'source_ids': [image['message_id']]}])
    response = client.post('/v1/memories/search', json=search_payload())
    assert response.status_code == 200
    data = response.json()
    validate_search(data, 100)
    assert len(data['data']) == 2
    assert data['data'][0]['content'][-1] == payload['messages'][0]['content'][-1]
    assert data['data'][1]['content'].endswith('Original related fact.')
    assert len(backend.rerank_inputs) == 1


@pytest.mark.parametrize('values', [
    {'association_count': 0}, {'association_count': 33},
    {'association_expand_count': 0}, {'association_expand_count': 6},
    {'association_count': True}, {'association_expand_count': 1.5},
])
def test_invalid_association_counts_are_rejected(settings, values):
    with pytest.raises(ValueError, match='association counts'):
        replace(settings, **values)


def test_association_settings_change_index_identity(settings):
    assert replace(settings, association_count=6).identity() != settings.identity()
    assert replace(settings, association_expand_count=1).identity() != settings.identity()
