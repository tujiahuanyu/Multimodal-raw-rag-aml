import pytest

from evaluation.src.aml.contract_smoke import validate_search


@pytest.mark.parametrize(
    'body,top_k',
    [
        ({}, 100),
        ({'data': {}}, 100),
        ({'data': [{'id': '', 'content': 'source'}]}, 100),
        ({'data': [{'id': 'memory', 'content': ''}]}, 100),
        ({'data': [{'id': 'memory', 'content': 'source', 'score': True}]}, 100),
        (
            {
                'data': [
                    {'id': 'first', 'content': 'source'},
                    {'id': 'second', 'content': 'source'},
                ]
            },
            1,
        ),
        (
            {
                'data': [
                    {'id': 'first', 'content': 'source', 'score': 0.1},
                    {'id': 'second', 'content': 'source', 'score': 0.9},
                ]
            },
            100,
        ),
        (
            {
                'data': [
                    {
                        'id': 'memory',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {'url': 'https://example.com/image.png'},
                            }
                        ],
                    }
                ]
            },
            100,
        ),
    ],
)
def test_response_validator_rejects_invalid_contract(body, top_k):
    with pytest.raises(ValueError):
        validate_search(body, top_k)
