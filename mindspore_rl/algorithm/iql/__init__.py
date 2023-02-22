# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
    Components for IQL.
"""
from mindspore_rl.algorithm.iql import config
from mindspore_rl.algorithm.iql.iql_session import IQLSession
from mindspore_rl.algorithm.iql.iql_trainer import IQLTrainer
from mindspore_rl.algorithm.iql.iql import IQLActor, IQLLearner, IQLPolicyAndNetwork

__all__ = ["config", "IQLSession", "IQLActor", "IQLLearner", "IQLPolicyAndNetwork", "IQLTrainer"]