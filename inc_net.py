import copy
import logging
import math
import torch
from torch import nn
import timm
from torch.nn import functional as F

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP=False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input
                #inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn,self.weight)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


def get_convnet(args, pretrained=False):

    name = args["convnet_type"].lower()
    #Resnet
    if name=="pretrained_resnet18":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model=resnet18(pretrained=False,args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet18-f37072fd.pth"),strict=False)
        return model.eval()
    elif name=="pretrained_resnet50":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model=resnet50(pretrained=False,args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet50-11ad3fa6.pth"),strict=False)
        return model.eval()
    elif name=="pretrained_resnet101":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model=resnet101(pretrained=False,args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet101-cd907fc2.pth"),strict=False)
        return model.eval()
    elif name=="pretrained_resnet152":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model=resnet152(pretrained=False,args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet152-f82ba261.pth"),strict=False)
        return model.eval()
    
    #NCM or NCM w/ Finetune
    elif name=="pretrained_vit_b16_224" or name=="vit_base_patch16_224":
        model=timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    elif name=="pretrained_vit_b16_224_in21k" or name=="vit_base_patch16_224_in21k":
        model=timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim=768
        return model.eval()
    
    # SSF 
    elif '_ssf' in name:
        if args["model_name"]=="ssf":
            from petl import vision_transformer_ssf #registers vit_base_patch16_224_ssf
            if name=="pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_ssf":
                model=timm.create_model("vit_base_patch16_224_in21k_ssf",pretrained=True, num_classes=0)
                model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # VPT
    elif '_vpt' in name:
        if args["model_name"]=="vpt":
            from petl.vpt import build_promptmodel
            if name=="pretrained_vit_b16_224_vpt":
                basicmodelname="vit_base_patch16_224" 
            elif name=="pretrained_vit_b16_224_in21k_vpt":
                basicmodelname="vit_base_patch16_224_in21k"
            
            print("modelname,",name,"basicmodelname",basicmodelname)
            VPT_type="Deep"
            #if args["vpt_type"]=='shallow':
            #    VPT_type="Shallow"
            Prompt_Token_num=5#args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname,  Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num=64#args["ffn_num"]
        if args["model_name"]=="adapter" :
            from petl import vision_transformer_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name=="pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name=="pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()


        print('This is for the BaseNet initialization.')
        self.convnet = get_convnet(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

class ResNetCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self._feature_dim=self.feature_dim

    def update_fc(self, nb_classes):
        fc = CosineLinear(self._feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self._feature_dim=self.feature_dim

    def update_fc(self, nb_classes):
        fc = CosineLinear(self._feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x)
        return out
