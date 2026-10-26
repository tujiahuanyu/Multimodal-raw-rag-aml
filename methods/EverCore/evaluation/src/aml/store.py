from contextlib import contextmanager
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

from .providers import ServiceError


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


class Store:
    def __init__(self, directory: Path, identity: str):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        self.path = directory / 'memory.sqlite3'
        self.lock = (directory / 'service.lock').open('a+b')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError(
                'Another service already uses this data directory'
            ) from None
        try:
            with self.connection() as connection:
                connection.executescript(
                    '''
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS requests (user_id TEXT, request_id TEXT, payload_hash TEXT NOT NULL, response TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(user_id,request_id));
                    CREATE TABLE IF NOT EXISTS messages (message_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL, ordinal INTEGER NOT NULL, content TEXT NOT NULL, render TEXT NOT NULL, role TEXT NOT NULL, timestamp INTEGER, UNIQUE(user_id,session_id,ordinal));
                    CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL, ordinal INTEGER NOT NULL, content TEXT NOT NULL, source_ids TEXT NOT NULL, vector BLOB NOT NULL, dimensions INTEGER NOT NULL);
                    CREATE INDEX IF NOT EXISTS chunks_user ON chunks(user_id);
                    CREATE TABLE IF NOT EXISTS media (user_id TEXT, hash TEXT, mime TEXT NOT NULL, data BLOB NOT NULL, PRIMARY KEY(user_id,hash));
                    CREATE TABLE IF NOT EXISTS captions (user_id TEXT, hash TEXT, text TEXT NOT NULL, PRIMARY KEY(user_id,hash));
                    CREATE TABLE IF NOT EXISTS embeddings (user_id TEXT, hash TEXT, vector BLOB NOT NULL, dimensions INTEGER NOT NULL, PRIMARY KEY(user_id,hash));
                '''
                )
                previous = connection.execute(
                    "SELECT value FROM metadata WHERE key='identity'"
                ).fetchone()
                if previous and previous['value'] != identity:
                    raise ValueError(
                        'Stored data uses a different model or configuration; select a new directory'
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (identity,)
                )
            os.chmod(self.path, 0o600)
        except Exception:
            self.close()
            raise

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA synchronous=FULL')
        try:
            yield connection
        finally:
            connection.close()

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()

    def replay(self, user, request, payload_hash):
        with self.connection() as connection:
            row = connection.execute(
                'SELECT payload_hash,response FROM requests WHERE user_id=? AND request_id=?',
                (user, request),
            ).fetchone()
        if row:
            if row['payload_hash'] != payload_hash:
                raise ServiceError(409, 'request_id_payload_conflict')
            return json.loads(row['response'])
        return None

    def messages(self, user, session):
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    'SELECT * FROM messages WHERE user_id=? AND session_id=? ORDER BY ordinal',
                    (user, session),
                )
            ]

    def has_user(self, user):
        with self.connection() as connection:
            return (
                connection.execute(
                    'SELECT 1 FROM chunks WHERE user_id=? LIMIT 1', (user,)
                ).fetchone()
                is not None
            )

    def caption(self, user, key, text=None):
        with self.connection() as connection:
            if text is not None:
                connection.execute(
                    'INSERT OR REPLACE INTO captions VALUES (?,?,?)', (user, key, text)
                )
            row = connection.execute(
                'SELECT text FROM captions WHERE user_id=? AND hash=?', (user, key)
            ).fetchone()
        return row['text'] if row else None

    def embedding(self, user, key, vector=None):
        with self.connection() as connection:
            if vector is not None:
                vector = np.asarray(vector, dtype='<f4')
                dimensions = str(len(vector))
                previous = connection.execute(
                    "SELECT value FROM metadata WHERE key='dimensions'"
                ).fetchone()
                if previous and previous['value'] != dimensions:
                    raise ServiceError(502, 'embedding_dimension_changed')
                connection.execute(
                    "INSERT OR IGNORE INTO metadata VALUES ('dimensions', ?)",
                    (dimensions,),
                )
                connection.execute(
                    'INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)',
                    (user, key, vector.tobytes(), len(vector)),
                )
            row = connection.execute(
                'SELECT vector,dimensions FROM embeddings WHERE user_id=? AND hash=?',
                (user, key),
            ).fetchone()
        return np.frombuffer(row['vector'], dtype='<f4').copy() if row else None

    def commit(self, request, payload_hash, messages, chunks, media):
        response = {
            'success': True,
            'request_id': request.request_id,
            'user_id': request.user_id,
            'session_id': request.session_id,
        }
        with self.connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                for row in messages:
                    connection.execute(
                        'INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)',
                        tuple(
                            row[name]
                            for name in [
                                'message_id',
                                'user_id',
                                'session_id',
                                'ordinal',
                                'content',
                                'render',
                                'role',
                                'timestamp',
                            ]
                        ),
                    )
                for key, (mime, data) in media.items():
                    connection.execute(
                        'INSERT OR IGNORE INTO media VALUES (?,?,?,?)',
                        (request.user_id, key, mime, data),
                    )
                connection.execute(
                    'DELETE FROM chunks WHERE user_id=? AND session_id=?',
                    (request.user_id, request.session_id),
                )
                connection.executemany(
                    'INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)',
                    [
                        (
                            row['chunk_id'],
                            request.user_id,
                            request.session_id,
                            position,
                            row['content'],
                            json.dumps(row['source_ids']),
                            np.asarray(row['vector'], dtype='<f4').tobytes(),
                            len(row['vector']),
                        )
                        for position, row in enumerate(chunks)
                    ],
                )
                connection.execute(
                    'INSERT INTO requests (user_id,request_id,payload_hash,response) VALUES (?,?,?,?)',
                    (
                        request.user_id,
                        request.request_id,
                        payload_hash,
                        json.dumps(response),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return response

    def candidates(self, user, vector, count):
        selected = []
        with self.connection() as connection:
            cursor = connection.execute(
                'SELECT chunk_id,session_id,content,source_ids,vector,dimensions FROM chunks WHERE user_id=?',
                (user,),
            )
            while rows := cursor.fetchmany(256):
                if any(row['dimensions'] != len(vector) for row in rows):
                    raise ServiceError(502, 'embedding_dimension_changed')
                matrix = np.stack(
                    [np.frombuffer(row['vector'], dtype='<f4') for row in rows]
                )
                scores = matrix @ vector
                for row, score in zip(rows, scores):
                    item = (
                        float(score),
                        row['chunk_id'],
                        row['session_id'],
                        row['content'],
                        row['source_ids'],
                    )
                    if len(selected) < count:
                        heapq.heappush(selected, item)
                    elif item > selected[0]:
                        heapq.heapreplace(selected, item)
        return [
            {
                'score': score,
                'chunk_id': chunk_id,
                'session_id': session,
                'content': content,
                'source_ids': json.loads(source_ids),
            }
            for score, chunk_id, session, content, source_ids in sorted(
                selected, reverse=True
            )
        ]

    def sources(self, user, identifiers):
        rows = []
        with self.connection() as connection:
            for identifier in identifiers:
                row = connection.execute(
                    'SELECT * FROM messages WHERE user_id=? AND message_id=?',
                    (user, identifier),
                ).fetchone()
                if row is None:
                    raise ServiceError(500, 'stored_source_missing')
                rows.append(dict(row))
        return rows

    def image(self, user, key):
        with self.connection() as connection:
            row = connection.execute(
                'SELECT mime,data FROM media WHERE user_id=? AND hash=?', (user, key)
            ).fetchone()
        if row is None:
            raise ServiceError(500, 'stored_image_missing')
        return row['mime'], row['data']

    def purge_user(self, user):
        with self.connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                for table in [
                    'requests',
                    'messages',
                    'chunks',
                    'media',
                    'captions',
                    'embeddings',
                ]:
                    connection.execute(f'DELETE FROM {table} WHERE user_id=?', (user,))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
