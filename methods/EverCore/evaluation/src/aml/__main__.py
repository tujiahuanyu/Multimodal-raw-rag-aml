import argparse
import os
from pathlib import Path

from .app import create_app
from .config import load_settings
from .store import Store


def main():
    parser = argparse.ArgumentParser(
        description='Run the participant-hosted Raw-RAG AML memory service'
    )
    parser.add_argument('command', choices=['serve', 'purge-user'])
    parser.add_argument('--env-file', type=Path, default=Path('.env.aml'))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--user-id')
    arguments = parser.parse_args()
    os.umask(0o077)
    settings = load_settings(arguments.env_file, arguments.data_dir)
    if arguments.command == 'purge-user':
        if not arguments.user_id:
            parser.error('--user-id is required')
        store = Store(settings.data_dir, settings.identity())
        try:
            store.purge_user(arguments.user_id)
        finally:
            store.close()
        print('User data deleted; other users were not changed')
    else:
        import uvicorn

        uvicorn.run(
            create_app(settings),
            host=arguments.host,
            port=arguments.port,
            workers=1,
            access_log=False,
        )


if __name__ == '__main__':
    main()
