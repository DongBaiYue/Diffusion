import torch
from torch import Tensor
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, ConfigMixin, ModelMixin
import json

# set seed
seed = 42
import numpy as np
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)

SD3_MODEL_PATH = '/workspace/codes/Diffusion/models/sd3-med'

LN_MEAN_MAP = {}
LN_VAR_MAP= {}
LN_MEAN_SIM_MAP = {}
LN_VAR_SIM_MAP = {}
LN_MEAN_REUSE_MAP = {}
LN_VAR_REUSE_MAP = {}

class ModuleConfig:
    def __init__(
        self,
        height: int = 1024,
        width: int = 1024,
        ln_simplify : bool = False
    ):
        self.height = height
        self.width = width
        self.ln_simplify = ln_simplify
        device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(device)
        self.device = device

class LayerNorm(torch.nn.Module):
    def __init__(self, module:torch.nn.LayerNorm, name:str, mean_reuse_list:list=None, var_reuse_list:list=None):
        super(LayerNorm, self).__init__()
        self.name = name
        self.mean_reuse_list = [False for _ in range(28)] if mean_reuse_list == None else mean_reuse_list
        self.var_reuse_list = [False for _ in range(28)] if var_reuse_list == None else var_reuse_list
        self.num = 0
        
        self.normalized_shape = module.normalized_shape
        self.weight = module.weight
        self.bias = module.bias
        self.eps = module.eps
        self.mean = None
        self.var = None

    def forward(self, input: Tensor) -> Tensor:
        # 计算均值和方差
        dim = [-i for i in range(1, len(self.normalized_shape)+1)]
        if not self.mean_reuse_list[self.num]:
            self.mean = input.mean(dim=dim, keepdim=True)
            LN_MEAN_MAP[self.name].append(round(self.mean[0][0].item(), 3))
        else:
            pass
        if not self.var_reuse_list[self.num]:
            variance = input.to(torch.float32).var(dim=dim, keepdim=True)
            self.var = torch.sqrt(variance + self.eps).to(torch.float16)
            LN_VAR_MAP[self.name].append(round(self.var[0][0].item(), 3))
        else:
            pass
        # 归一化  
        normalized_input = (input - self.mean) / self.var
        # 应用权重和偏置
        if self.weight is not None:
            normalized_input = normalized_input * self.weight
        if self.bias is not None:
            normalized_input = normalized_input + self.bias
        self.num += 1
        return normalized_input
        # return torch.nn.functional.layer_norm(
        #     input, self.normalized_shape, self.weight, self.bias, self.eps)

class transformerLN(ModelMixin, ConfigMixin):
    def __init__(self, model: SD3Transformer2DModel, module_config: ModuleConfig):
        assert isinstance(model, SD3Transformer2DModel)
        super(transformerLN, self).__init__()
        if module_config.ln_simplify:
            with open('mean_reuse.json', 'r') as file:
                LN_MEAN_REUSE_MAP = json.load(file)
            with open('var_reuse.json', 'r') as file:
                LN_VAR_REUSE_MAP = json.load(file)
        for name, module in model.named_modules():
            for subname, submodule in module.named_children():
                if isinstance(submodule, torch.nn.LayerNorm):
                    layer_name = name + "." + subname
                    LN_MEAN_MAP[layer_name] = []
                    LN_VAR_MAP[layer_name] = []
                    if module_config.ln_simplify:
                        mean_reuse_list = LN_MEAN_REUSE_MAP[layer_name]
                        var_reuse_list = LN_VAR_REUSE_MAP[layer_name]
                    else:
                        mean_reuse_list = None
                        var_reuse_list = None
                    wrapped_submodule = LayerNorm(submodule, layer_name, mean_reuse_list, var_reuse_list)
                    setattr(module, subname, wrapped_submodule)
            
        self.model = model
        self.module_config = module_config
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    @property
    def config(self):
        return self.model.config


class SD3Pipeline:
    def __init__(self, pipeline: StableDiffusion3Pipeline, module_config: ModuleConfig):
        self.pipeline = pipeline
        self.module_config = module_config

        self.static_inputs = None
        self.prepare()
    
    @staticmethod
    def from_pretrained(module_config: ModuleConfig, **kwargs):
        device = module_config.device
        pretrained_model_name_or_path = SD3_MODEL_PATH
        torch_dtype = torch.float16
        transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="transformer"
        ).to(device)

        transformer = transformerLN(transformer, module_config)
        
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, transformer=transformer, **kwargs
        ).to(device)

        return SD3Pipeline(pipeline, module_config)
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        module_config = self.module_config
        return self.pipeline(height=module_config.height, width=module_config.width, *args, **kwargs)
    @torch.no_grad()
    def prepare(self, **kwargs):
        pass

def mean_var_analysis():
    # write to json
    with open('mean1.json', 'w') as file:
        json.dump(LN_MEAN_MAP, file, indent=4)
    with open('var1.json', 'w') as file:
        json.dump(LN_VAR_MAP, file, indent=4)

def compute_similarity_mean_num(sim_range:float=0.01):
    with open('mean.json', 'r') as file:
        LN_MEAN_MAP = json.load(file)
    total_ln_num = 0
    sim_ln_num = 0
    for layer_name, means in LN_MEAN_MAP.items():
        similar_tuples = []
        mean_pre_index = 0
        for mean_now_index in range(1, len(means)):
            mean_pre = means[mean_pre_index]
            mean_now = means[mean_now_index]
            if(abs(mean_now-mean_pre) > sim_range):
                if(mean_now_index-mean_pre_index >= 2):
                    similar_tuples.append((mean_pre_index, mean_now_index-1))
                    sim_ln_num += mean_now_index-1- mean_pre_index
                mean_pre_index = mean_now_index
            elif(mean_now_index==len(means)-1):
                similar_tuples.append((mean_pre_index, mean_now_index))
                sim_ln_num += mean_now_index- mean_pre_index
            total_ln_num += 1
        LN_MEAN_SIM_MAP[layer_name] = similar_tuples
    with open('mean_sim.json', 'w') as file:
        json.dump(LN_MEAN_SIM_MAP, file, indent=4)
    print(f'total_ln_num: {total_ln_num}')
    print(f'sim_ln_num: {sim_ln_num}')

def compute_similarity_var_num(sim_range:float=0.01):
    with open('var.json', 'r') as file:
        LN_VAR_MAP = json.load(file)
    total_ln_num = 0
    sim_ln_num = 0
    for layer_name, vars in LN_VAR_MAP.items():
        similar_tuples = []
        mean_pre_index = 0
        for mean_now_index in range(1, len(vars)):
            mean_pre = vars[mean_pre_index]
            mean_now = vars[mean_now_index]
            if(abs(mean_now-mean_pre) > sim_range):
                if(mean_now_index-mean_pre_index >= 2):
                    similar_tuples.append((mean_pre_index, mean_now_index-1))
                    sim_ln_num += mean_now_index-1- mean_pre_index
                mean_pre_index = mean_now_index
            elif(mean_now_index==len(vars)-1):
                similar_tuples.append((mean_pre_index, mean_now_index))
                sim_ln_num += mean_now_index- mean_pre_index
            total_ln_num += 1
        LN_VAR_SIM_MAP[layer_name] = similar_tuples
    with open('var_sim.json', 'w') as file:
        json.dump(LN_VAR_SIM_MAP, file, indent=4)
    print(f'total_ln_num: {total_ln_num}')
    print(f'sim_ln_num: {sim_ln_num}')

def compute_reuse_mean_lists():
    with open('mean_sim.json', 'r') as file:
        LN_MEAN_SIM_MAP = json.load(file)
    for layer_name, similar_tuples in LN_MEAN_SIM_MAP.items():
        reuse_list = [False for _ in range(28)]
        for similar_tuple in similar_tuples:
            for index in range(similar_tuple[0]+1, similar_tuple[1]+1):
               reuse_list[index] = True
        LN_MEAN_REUSE_MAP[layer_name] = reuse_list
    with open('mean_reuse.json', 'w') as file:
        json.dump(LN_MEAN_REUSE_MAP, file, indent=4)

def compute_reuse_var_lists():
    with open('var_sim.json', 'r') as file:
        LN_VAR_SIM_MAP = json.load(file)
    for layer_name, similar_tuples in LN_VAR_SIM_MAP.items():
        reuse_list = [False for _ in range(28)]
        for similar_tuple in similar_tuples:
            for index in range(similar_tuple[0]+1, similar_tuple[1]+1):
               reuse_list[index] = True
        LN_VAR_REUSE_MAP[layer_name] = reuse_list
    with open('var_reuse.json', 'w') as file:
        json.dump(LN_VAR_REUSE_MAP, file, indent=4)
        
                
def run_sd3(ln_simplify:bool=False):
    module_config = ModuleConfig(width=1024,height=1024, ln_simplify=ln_simplify)
    pipeline = SD3Pipeline.from_pretrained(module_config=module_config)
    # PROMPT = 'Astronaut in a jungle, cold color palette, muted colors, detailed, 8k'
    PROMPT = 'A kitten holding hello world'
    image = pipeline(
        prompt=PROMPT,
        generator=torch.Generator(device="cuda")
    ).images[0]
    image.save('output_pictures/cat_sim.png')

# compute_similarity_var_num(sim_range=0.01)
# compute_reuse_var_lists()
run_sd3(ln_simplify=True)