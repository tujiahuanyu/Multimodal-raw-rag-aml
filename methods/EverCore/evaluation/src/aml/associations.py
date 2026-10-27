import heapq
import json

import numpy as np

from .providers import ServiceError
from .store import digest


def refresh_links(connection, user, new_identifiers, count):
    nodes = {}
    for row in connection.execute(
        'SELECT chunk_id,source_ids,vector FROM chunks WHERE user_id=?', (user,)
    ):
        nodes[row['chunk_id']] = {
            'source_ids': json.loads(row['source_ids']),
            'vectors': [np.frombuffer(row['vector'], dtype='<f4')],
        }
    for row in connection.execute(
        'SELECT message_id,image_hash,vector FROM image_views WHERE user_id=? ORDER BY ordinal',
        (user,),
    ):
        identifier = digest(('image', user, row['message_id'], row['image_hash']))
        node = nodes.setdefault(
            identifier, {'source_ids': [row['message_id']], 'vectors': []}
        )
        node['vectors'].append(np.frombuffer(row['vector'], dtype='<f4'))
    identifiers = sorted(nodes)
    limit = min(count, max(0, len(identifiers) - 1))
    neighbors = {identifier: {} for identifier in identifiers}
    for row in connection.execute(
        'SELECT source_id,target_id,similarity FROM chunk_links WHERE user_id=?',
        (user,),
    ):
        if row['source_id'] in nodes and row['target_id'] in nodes:
            neighbors[row['source_id']][row['target_id']] = row['similarity']
    new_identifiers = set(new_identifiers) & nodes.keys()
    rebuild = new_identifiers | {
        identifier for identifier in identifiers if len(neighbors[identifier]) != limit
    }
    if limit and rebuild:
        vectors, target_indices, positions = [], [], {}
        for number, identifier in enumerate(identifiers):
            positions[identifier] = list(
                range(len(vectors), len(vectors) + len(nodes[identifier]['vectors']))
            )
            vectors.extend(nodes[identifier]['vectors'])
            target_indices.extend([number] * len(nodes[identifier]['vectors']))
        if len({len(vector) for vector in vectors}) != 1:
            raise ServiceError(502, 'embedding_dimension_changed')
        matrix = np.stack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms == 0).any():
            raise ServiceError(502, 'association_invalid_vector')
        matrix = matrix / norms
        target_indices = np.asarray(target_indices)
        sources = sorted(rebuild)
        for start in range(0, len(sources), 32):
            batch = sources[start : start + 32]
            columns = [position for source in batch for position in positions[source]]
            similarities = matrix @ matrix[columns].T
            offset = 0
            for source in batch:
                end = offset + len(positions[source])
                scores = np.full(len(identifiers), -np.inf, dtype=np.float32)
                np.maximum.at(scores, target_indices, similarities[:, offset:end].max(axis=1))
                offset = end
                neighbors[source] = dict(
                    (target, score)
                    for score, target in heapq.nlargest(
                        limit,
                        (
                            (float(score), target)
                            for target, score in zip(identifiers, scores)
                            if target != source
                        ),
                    )
                )
                if source in new_identifiers:
                    for target, score in zip(identifiers, scores):
                        if target in rebuild:
                            continue
                        choices = neighbors[target]
                        choices[source] = float(score)
                        neighbors[target] = dict(
                            (identifier, value)
                            for value, identifier in heapq.nlargest(
                                limit, ((value, identifier) for identifier, value in choices.items())
                            )
                        )
    connection.execute('DELETE FROM chunk_links WHERE user_id=?', (user,))
    connection.executemany(
        'INSERT INTO chunk_links VALUES (?,?,?,?,?)',
        [
            (
                user,
                source,
                target,
                float(similarity),
                json.dumps(nodes[target]['source_ids']),
            )
            for source, targets in neighbors.items()
            for target, similarity in targets.items()
            if limit
        ],
    )
