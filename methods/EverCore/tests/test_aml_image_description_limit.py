import pytest

from evaluation.src.aml.providers import ServiceError, image_descriptions


@pytest.mark.parametrize(
    'value,expected',
    [
        ([], []),
        ([' one '], ['one']),
        (['one', 'two', 'three', 'four'], ['one', 'two', 'three', 'four']),
        (
            ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
            ['one', 'two', 'three', 'four'],
        ),
        (
            ['one', ' ONE ', 'two', 'TWO', 'three', 'four', 'five', 'six', 'seven'],
            ['one', 'two', 'three', 'four'],
        ),
        ([' Same  description ', 'same description'] * 5, ['Same  description']),
    ],
)
def test_description_limit_is_applied_after_existing_deduplication(value, expected):
    assert image_descriptions(value) == expected
    assert image_descriptions(image_descriptions(value)) == expected


@pytest.mark.parametrize(
    'value',
    [
        None, 'one', {'descriptions': []}, ('one',),
        [None], [''], [' '], ['x' * 1201],
        ['one', 'two', 'three', 'four', None],
        ['one', 'two', 'three', 'four', ' '],
        ['one', 'two', 'three', 'four', 'x' * 1201],
    ],
)
def test_invalid_items_are_still_rejected_even_beyond_the_limit(value):
    with pytest.raises(ServiceError) as caught:
        image_descriptions(value)
    assert caught.value.status == 502
    assert caught.value.code == 'vision_invalid_image_details'
