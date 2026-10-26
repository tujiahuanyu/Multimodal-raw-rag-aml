from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TextPart(StrictModel):
    type: Literal['text']
    text: Annotated[StrictStr, Field(min_length=1)]


class ImageURL(StrictModel):
    url: StrictStr
    detail: Literal['auto', 'low', 'high'] | None = None


class ImagePart(StrictModel):
    type: Literal['image_url']
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator='type')]
Content = (
    Annotated[StrictStr, Field(min_length=1)]
    | Annotated[list[ContentPart], Field(min_length=1)]
)
Identity = Annotated[StrictStr, Field(min_length=1, max_length=2048)]


class MessageInput(StrictModel):
    role: Literal['user', 'assistant']
    content: Content
    timestamp: StrictInt | None = None

    @field_validator('timestamp')
    @classmethod
    def valid_timestamp(cls, value):
        if value is not None:
            try:
                datetime.fromtimestamp(value / 1000, timezone.utc)
            except (ValueError, OverflowError, OSError) as exception:
                raise ValueError('Invalid Unix millisecond timestamp') from exception
        return value


class AddRequest(StrictModel):
    request_id: Identity
    user_id: Identity
    session_id: Identity
    messages: Annotated[list[MessageInput], Field(min_length=1)]


class SearchRequest(StrictModel):
    user_id: Identity
    query: Content
    options: list[StrictStr] | None = None
    top_k: Annotated[StrictInt, Field(ge=0, le=1000)]
