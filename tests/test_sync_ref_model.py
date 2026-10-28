# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from trl import DPOConfig, GRPOConfig, KTOConfig, RLOOConfig


@pytest.mark.parametrize("config_cls", [DPOConfig, GRPOConfig, KTOConfig, RLOOConfig])
@pytest.mark.parametrize("invalid_steps", [0, -1, -512])
def test_sync_ref_model_non_positive_steps_raises(tmp_path, config_cls, invalid_steps):
    with pytest.raises(ValueError, match="`ref_model_sync_steps` must be greater than 0"):
        config_cls(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            sync_ref_model=True,
            ref_model_sync_steps=invalid_steps,
        )


@pytest.mark.parametrize("config_cls", [DPOConfig, GRPOConfig, KTOConfig, RLOOConfig])
def test_sync_ref_model_positive_steps_succeeds(tmp_path, config_cls):
    config = config_cls(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        sync_ref_model=True,
        ref_model_sync_steps=256,
    )
    assert config.sync_ref_model is True
    assert config.ref_model_sync_steps == 256


@pytest.mark.parametrize("config_cls", [DPOConfig, GRPOConfig, KTOConfig, RLOOConfig])
def test_sync_ref_model_disabled_by_default(tmp_path, config_cls):
    config = config_cls(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
    )
    assert config.sync_ref_model is False
    assert config.ref_model_sync_steps == 512
