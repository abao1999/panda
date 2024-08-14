from typing import Union, Sequence
from chronos_dysts.tokenizer import ChronosTokenizer, ChronosConfig
from chronos_dysts.model import ChronosModel

# float or sequence (list-like) of floats
FloatOrFloatSequence = Union[float, Sequence[float]]

# alias for tokenizer base class
ChronosTokenizerType = ChronosTokenizer
ChronosModelType = ChronosModel
ChronosConfigType = ChronosConfig