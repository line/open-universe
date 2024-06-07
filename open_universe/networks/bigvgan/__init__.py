# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software. 
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The code in this sub-package was taken from the BIGVGAN repository
https://github.com/NVIDIA/BigVGAN

Most of it is under MIT License, but some parts are different. Please
consult the orginal repository for all the details.
"""
from .alias_free_act import Activation1d
from .gan import (
    DiscriminatorP,
    DiscriminatorR,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from .snake import AliasFreeSnake, Snake, SnakeBeta
