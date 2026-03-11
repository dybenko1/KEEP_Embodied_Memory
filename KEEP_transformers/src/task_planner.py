import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from torch.nn import CrossEntropyLoss
import guidance
# from guidance import models, gen,select
# from guidance._program import Program
import logging
from sentence_transformers import SentenceTransformer
import copy
import os
from PIL import Image
import time, datetime
import matplotlib
import time
# 使用非交互式后端
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt
# import torch.nn.functional as F

recompute = True
rm_first_kv_token = False
need_load_kv = False
if rm_first_kv_token:
    start_1_length = 1
else:
    start_1_length = 0

class message_kv:
    def __init__(self, content):
        self.content = content
        self.device = 'cuda:0'
        self.kv = {}
        self.semantic_vector = None
        self.length = 0
        self.id = None
        self.token_ids = None
        self.kv_place = None
        

    def move_kv_to_device(self, device):
        self.device = device
        return tuple(tuple((t[0].to(self.device),t[1].to(self.device))) for t in self.kv)

    def get_semantic_vector(self, model):
        if self.semantic_vector is None:
            self.semantic_vector = model.encode(str(self.content))


    
    def get_kv(self, model,tokenizer,position_embedding_place=0):
        prompt_tokens = tokenizer(str(self.content)+'\n', add_special_tokens=True, return_tensors="pt", padding=True).to(model.device)
        self.token_ids = prompt_tokens["input_ids"]
        seq_length = prompt_tokens["input_ids"].shape[1]
        self.kv_place = [position_embedding_place, position_embedding_place+seq_length]
        with torch.no_grad():
            prompt_output = model(
                **prompt_tokens, 
                use_cache=True, 
                )
        if need_load_kv:
            self.kv = tuple((torch.tensor(tensor[0]).to('cpu'),torch.tensor(tensor[1]).to('cpu')) for tensor in prompt_output.past_key_values)
        else:
            self.kv = prompt_output.past_key_values
        self.length = self.kv[0][0].shape[2]  # assuming all layers have the same length


        

class overall_kv_schedule:
    def __init__(self,semantic_model,planner_model,tokenizer):
        self.semantic_model = semantic_model
        self.planner_model = planner_model
        self.tokenizer = tokenizer
        self.static_prompt = None
        self.static_kv = None
        self.kv_capacity = 10                    # 2:14    # 5:16    
        self.messages_objects = []
        self.messages_examples = []
        self.similarity_objects = []
        self.similarity_examples = []
        self.position_embedding_place_objects = 0
        self.position_embedding_place_examples = 4000
        self.query = None
        self.query_semantic_vector = None
        self.kv_places = []
        self.memory_num = 0


    def reset(self):
        self.messages_objects = []
        self.messages_examples = []
        self.similarity_objects = []
        self.similarity_examples = []
        if self.static_kv != None:
            self.position_embedding_place_objects = self.static_kv[0][0].shape[2]
        else:
            self.position_embedding_place_objects = 0
        self.position_embedding_place_examples = 4000
        self.query = None
        self.query_semantic_vector = None
        self.kv_places = []


    def use_static_memory(self,messages):
        prompts = []
        token_ids = []
        positions = []
        position = 0
        for j in range(len(messages)):
            message = messages[j]
            task_desc = message["task description"].strip()
            if task_desc[-1].isalnum():
                task_desc += '.'
            task_desc = task_desc.capitalize()
            prompt = f'Human: {task_desc}' + '\n'
            prompt += 'Robot: '
            last_i = 0
            for i, step in enumerate(message['NL steps']):
                prompt += f'{i+1}. {step}, '
                last_i = i+1
            prompt += f'{last_i+1}. done.'
            prompts.append(prompt)
            token_id = self.tokenizer(str(prompt)+'\n', add_special_tokens=True, return_tensors="pt", padding=True).to(self.planner_model.device)
            token_ids.append(token_id["input_ids"])
            positions.append([position,position+token_id["input_ids"].shape[1]])
            position = position+token_id["input_ids"].shape[1]
        with torch.no_grad():
            prompt_output = self.planner_model(torch.cat(token_ids,dim=1), use_cache=True, )
        for j in range(len(messages)):
            self.messages_examples.append(message_kv(prompts[j]))
            self.messages_examples[-1].get_semantic_vector(self.semantic_model)
            self.messages_examples[-1].token_ids = token_ids[j]
            self.messages_examples[-1].length = token_ids[j].shape[1]
            self.messages_examples[-1].kv_place = [self.position_embedding_place_examples, self.position_embedding_place_examples+token_ids[j].shape[1]]
            self.messages_examples[-1].kv = tuple((torch.tensor(tensor[0])[:,:,positions[j][0]:positions[j][1]],torch.tensor(tensor[1])[:,:,positions[j][0]:positions[j][1]]) for tensor in prompt_output.past_key_values)
            self.position_embedding_place_examples += self.messages_examples[-1].length
            self.similarity_examples.append(self.semantic_model.similarity(self.query_semantic_vector, self.messages_examples[-1].semantic_vector)[0][0])


    def move_kv_layer_wise(self,example_num):
        for j in range(1,48):
            # start = time.time()
            for i in range(example_num):
                if i == 0 :
                    kv_tomove = tuple(tuple((t[0].to("cuda:0"),t[1].to("cuda:0"))) for t in self.messages_examples[i].kv[:j])
                else:
                    kv_tomove = tuple(tuple((torch.cat((t1[0], t2[0].to("cuda:0")), dim=2),torch.cat((t1[1], t2[1].to("cuda:0")), dim=2)))for t1, t2 in zip(kv_tomove,self.messages_examples[i%6].kv[:j]))
            for i in range(len(self.messages_objects)):
                kv_tomove = tuple(tuple((torch.cat((t1[0], t2[0].to("cuda:0")), dim=2),torch.cat((t1[1], t2[1].to("cuda:0")), dim=2)))for t1, t2 in zip(kv_tomove,self.messages_objects[i].kv[:j]))
            


    def get_similarity(self,sentences):
        embeddings = self.semantic_model.encode(sentences)
        similarities = self.semantic_model.similarity(embeddings[0], embeddings)
        return similarities[0][1:]

    def get_query_semantic(self, query):
        self.query_semantic_vector = self.semantic_model.encode(query)
    
    def get_static_kv(self,message):
        self.static_prompt = message
        prompt_tokens = self.tokenizer(message, add_special_tokens=True, return_tensors="pt", padding=True).to(self.planner_model.device)
        seq_length = prompt_tokens["input_ids"].shape[1]
        custom_positions = torch.arange(self.position_embedding_place_objects, self.position_embedding_place_objects+seq_length).unsqueeze(0).to(self.planner_model.device)
        with torch.no_grad():
            prompt_output = self.planner_model(**prompt_tokens, use_cache=True, position_ids=custom_positions)
        self.static_kv = prompt_output.past_key_values 
        self.position_embedding_place_objects += self.static_kv[0][0].shape[2]

    def add_message_examples(self, message):
        task_desc = message["task description"].strip()
        if task_desc[-1].isalnum():
            task_desc += '.'
        task_desc = task_desc.capitalize()
        prompt = f'Human: {task_desc}' + '\n'
        prompt += 'Robot: '
        last_i = 0
        for i, step in enumerate(message['NL steps']):
            prompt += f'{i+1}. {step}, '
            last_i = i+1
        prompt += f'{last_i+1}. done.'
        self.messages_examples.append(message_kv(prompt))
        self.messages_examples[-1].get_semantic_vector(self.semantic_model)
        self.messages_examples[-1].get_kv(self.planner_model,self.tokenizer,self.position_embedding_place_examples)
        self.position_embedding_place_examples += self.messages_examples[-1].length
        self.similarity_examples.append(self.semantic_model.similarity(self.query_semantic_vector, self.messages_examples[-1].semantic_vector)[0][0])

    def add_message_objects(self, message):
        self.messages_objects.append(message_kv(message))
        self.messages_objects[-1].get_semantic_vector(self.semantic_model)
        self.messages_objects[-1].get_kv(self.planner_model,self.tokenizer,self.position_embedding_place_objects)
        self.kv_places.append([self.position_embedding_place_objects,self.position_embedding_place_objects+self.messages_objects[-1].length])
        self.position_embedding_place_objects += self.messages_objects[-1].length
        self.similarity_objects.append(self.semantic_model.similarity(self.query_semantic_vector, self.messages_objects[-1].semantic_vector)[0][0])

    def add_message_objects_by_observation(self,obj,env):
        environment_objects = [objects['name'][:objects['name'].find('_')] for objects in env.last_event.metadata['objects']]
        similarity = self.get_similarity([obj['object']] + environment_objects)
        max_similarity = max(similarity)
        objects_ids = [i for i,x in enumerate(similarity) if x == max_similarity]
        if len(objects_ids) > 1:
            distances = [env.last_event.metadata['objects'][i]['distance'] for i in objects_ids]
            min_distance = min(distances)
            objects_ids = [objects_ids[i] for i,x in enumerate(distances) if x == min_distance]
        real_object = env.last_event.metadata['objects'][objects_ids[0]]['name']
        obj['object'] = real_object[:real_object.find('_')]
        # obj['objectId'] = real_object[real_object.find('_')+1:]
        obj_id = env.last_event.metadata['objects'][objects_ids[0]]['objectId']
        found_obj_id = [message.id for message in self.messages_objects]
        if obj_id not in found_obj_id:
            self.add_message_objects(obj)
            self.messages_objects[-1].id = obj_id
        else:
            obj_idx = found_obj_id.index(obj_id)
            message = self.messages_objects[obj_idx]
            # 'state': 'sliced', 'position': 'in the fridge'
            if not obj == message.content:
            # if not (obj['state'] == message.content['state'] and obj['position'] == message.content['position']):
                del self.kv_places[self.kv_places.index(self.messages_objects[obj_idx].kv_place)]
                del self.messages_objects[obj_idx]
                del self.similarity_objects[obj_idx]
                self.add_message_objects(obj)
                self.messages_objects[-1].id = obj_id

    def add_message_objects_by_action(self,env,obj_id,message1,message2):
        environment_objects = [objects['objectId'] for objects in env.last_event.metadata['objects']]
        _obj = env.last_event.metadata['objects'][environment_objects.index(obj_id)]
        obj = {
            'object': _obj['name'][:_obj['name'].find('_')],
            'state': message1,
            'position': message2,
            # 'objectId': _obj['objectId'],
        }
        found_obj_id = [message.id for message in self.messages_objects]
        if obj_id not in found_obj_id:
            self.add_message_objects(obj)
            self.messages_objects[-1].id = obj_id
        else:
            obj_idx = found_obj_id.index(obj_id)
            message = self.messages_objects[obj_idx]
            # 'state': 'sliced', 'position': 'in the fridge'
            if not obj == message.content:
            # if not (obj['state'] == message.content['state'] and obj['position'] == message.content['position']):
                del self.kv_places[self.kv_places.index(self.messages_objects[obj_idx].kv_place)]
                del self.messages_objects[obj_idx]
                del self.similarity_objects[obj_idx]
                self.add_message_objects(obj)
                self.messages_objects[-1].id = obj_id


    def get_kv(self):
        module_lengths = []
        token_ids = None

        prompt = self.static_prompt
        kv = self.static_kv
        module_lengths.append([0,self.static_kv[0][0].shape[2]])
        
        sorted_indices = sorted(range(len(self.similarity_objects)), key=lambda i: self.similarity_objects[i], reverse=True)

        object_kv = []
        example_kv = []
        if need_load_kv:
            start = time.time()
            for i in range(min(self.kv_capacity,len(self.similarity_objects))):
                index = sorted_indices[i]
                object_kv.append(self.messages_objects[index].move_kv_to_device('cuda:0'))
            for i in range(len(self.messages_examples)):
                example_kv.append(self.messages_examples[i].move_kv_to_device('cuda:0'))
        else:
            for i in range(min(self.kv_capacity,len(self.similarity_objects))):
                index = sorted_indices[i]
                object_kv.append(self.messages_objects[index].kv)
            for i in range(len(self.messages_examples)):
                example_kv.append(self.messages_examples[i].kv)

        for i in range(min(self.kv_capacity,len(self.similarity_objects))):
            index = sorted_indices[i]
            module_lengths.append([module_lengths[-1][-1],module_lengths[-1][-1]+self.messages_objects[index].length-start_1_length])
            if token_ids is None:
                token_ids = self.messages_objects[index].token_ids[:,start_1_length:]
            else:
                token_ids=torch.cat((token_ids,self.messages_objects[index].token_ids[:,start_1_length:]),dim=1)
            kv = tuple(tuple((torch.cat((t1[0], t2[0]), dim=2),torch.cat((t1[1][:,:,start_1_length:], t2[1][:,:,start_1_length:]), dim=2)))for t1, t2 in zip(kv,object_kv[i]))
            prompt += str(self.messages_objects[index].content) +'\n'
        for i in range(len(self.messages_examples)):
            module_lengths.append([module_lengths[-1][-1],module_lengths[-1][-1]+self.messages_examples[i].length-start_1_length])
            if token_ids is None:
                token_ids = self.messages_examples[i].token_ids[:,start_1_length:]
            else:
                token_ids=torch.cat((token_ids,self.messages_examples[i].token_ids[:,start_1_length:]),dim=1)
            prompt += self.messages_examples[i].content +'\n'
            kv = tuple(tuple((torch.cat((t1[0], t2[0]), dim=2),torch.cat((t1[1][:,:,start_1_length:], t2[1][:,:,start_1_length:]), dim=2)))for t1, t2 in zip(kv,example_kv[i]))
        if recompute:
            self.planner_model.model.module_lengths = module_lengths
            self.planner_model.model.token_ids = token_ids
        self.memory_num += len(module_lengths) * self.planner_model.config.num_hidden_layers
        return kv,prompt,module_lengths


    def recompute_kv_num(self,decay_ratio,mem_num,layer_num):
        # decay_ratio = 0.85

        recompute_num = mem_num
        total_recompute_num = 0
        for i in range(layer_num):
            total_recompute_num += max(1,int(recompute_num))
            recompute_num *= decay_ratio

        return total_recompute_num


        




class TaskPlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.planner.device
        self.use_object = cfg.planner.use_object
        self.max_steps = cfg.planner.max_steps
        self.model_name = cfg.planner.model_name
        self.scoring_batch_size = cfg.planner.scoring_batch_size
        self.score_function = cfg.planner.score_function
        self.scoring_mode = cfg.planner.scoring_mode
        self.use_predefined_prompt = cfg.planner.use_predefined_prompt
        self.memory = []
        self.use_memory = cfg.planner.use_memory
        self.use_kv = cfg.planner.use_kv
        local_model_path = "PATH/TO/SENTENCE_TRANSFORMER"  # 指向包含模型文件的本地目录
        # self.tokenizer_clip = CLIPTokenizer.from_pretrained(local_model_path)
        self.model_clip = SentenceTransformer(local_model_path).to("cuda:0")
        self.memory_select = None
        self.example_emb = None
        
        

        # Load pre-trained model
        print(f"Loading LLM and tokenizer: {self.model_name}")

        model_args = {'pretrained_model_name_or_path': self.model_name, 
                      'torch_dtype': torch.float16}
        if cfg.planner.use_accelerate_device_map:
            model_args['device_map'] = "auto"
        if cfg.planner.load_in_8bit:
            model_args['load_in_8bit'] = True
        if cfg.planner.load_in_4bit:
            model_args['load_in_4bit'] = True
        model_args['use_auth_token'] = cfg.planner.hf_auth_token

        if cfg.planner.scoring_mode == 'guidance':
            model_args.pop('pretrained_model_name_or_path')
            tokenizer = None
            if "OpenAI" in self.model_name:
                openai_model_name = self.model_name.split('/')[1]
                guidance.llm = guidance.llms.OpenAI(openai_model_name, api_key=cfg.planner.openai_api_key)
            else:
                if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name:
                    tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
                if "bigscience/bloom" == self.model_name:  # bloom 175B
                    model_args['max_memory'] = {0: '60GB', 1: '80GB', 2: '48GB', 3: '48GB', 4: '48GB'}

                model_args['max_memory'] = {1: '47GB', 0: '0GB'}
                self.llm = models.Transformers(self.model_name,do_sample=False,trust_remote_code=True,**model_args)


            self.model = None
            self.tokenizer = None

            logging.getLogger("guidance").setLevel(logging.WARNING)

        else:
            if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name:  # these do not work well with automodel
                self.model = LlamaForCausalLM.from_pretrained(**model_args)
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**model_args)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if not cfg.planner.use_accelerate_device_map and not cfg.planner.load_in_8bit and not cfg.planner.load_in_4bit:
                self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer.pad_token_id = 0
            print(f"Loading done\n")

        if self.use_kv:
            self.kv_schedule = overall_kv_schedule(self.model_clip, self.model,self.tokenizer)
        # Load prompt
        self.prompt = self.init_prompt(cfg)

    def get_similarity(self,sentences):
        embeddings = self.model_clip.encode(sentences)
        similarities = self.model_clip.similarity(embeddings[0], embeddings)
        return similarities[0][1:]

    def reset(self, nl_act_list, nl_obj_list):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)

    def reset(self):
        self.skill_set = self.init_skill_set()

    def init_prompt(self, cfg):
        raise NotImplementedError()

    def init_skill_set(self, nl_act_list, nl_obj_list):
        raise NotImplementedError()

    def update_skill_set(self, previous_step, nl_obj_list):
        raise NotImplementedError()

    def score(self, prompt, skill_set,step_num,query):
        # print(f"Scoring skill set: {skill_set}")
        
        scores = {}
        batch_skill_set_list = [skill_set[chunk:chunk + self.scoring_batch_size] for chunk in
                                range(0, len(skill_set), self.scoring_batch_size)]

        if self.scoring_mode == 'guidance':
            print(f"Prompt: {prompt}")

            with guidance.llm_context(llm=self.llm, kv_cache=None):
                out = self.llm + prompt + select(name='step', options=list(skill_set))
            score = out["step"].strip()


        elif self.scoring_mode == 'reuse_prompt' or self.scoring_mode == 'naive':
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True)
            if not self.cfg.planner.use_accelerate_device_map:
                prompt_tokens = prompt_tokens.to(self.model.device)
            prompt_len = prompt_tokens.attention_mask[0].sum().item()

            ## ==================================================
            ## get past_kv
            ## ===================================================
            with torch.no_grad():
                if self.use_kv:
                    kv,final_prompt,module_length = self.kv_schedule.get_kv()
                    prompt_output = self.model(input_ids=prompt_tokens.input_ids,past_key_values=kv, use_cache=True)
                    kv= prompt_output.past_key_values
                else:
                    prompt_output = self.model(**prompt_tokens, use_cache=True)
                    kv = prompt_output.past_key_values



            for batch_skill_set in batch_skill_set_list:
                batch_sentence = [f"{prompt} {skill}" for skill in batch_skill_set]
                size_B = len(batch_skill_set)
                if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name:
                    batch_skill_set_for_model = batch_skill_set
                else:
                    batch_skill_set_for_model = [f" {skill}" for skill in batch_skill_set]

                with torch.no_grad():
                    if self.scoring_mode == 'reuse_prompt':
                            skill_tokens = self.tokenizer(batch_skill_set_for_model, add_special_tokens=False,
                                                          return_tensors="pt", padding=True)
                            if not self.cfg.planner.use_accelerate_device_map:
                                skill_tokens = skill_tokens.to(self.model.device)

                            concat_attention_mask = torch.cat(
                                (torch.ones(1,kv[0][0].shape[2]).to(self.model.device).repeat(size_B, 1),prompt_tokens.attention_mask.repeat(size_B, 1), skill_tokens.attention_mask), dim=1)

                            ###!!!!! change this !!!!!!
                            
                            batch_past_key_values = self.duplicate_past_key_values(kv, size_B)


                            seq_length = skill_tokens["input_ids"].shape[1]
                            if self.use_kv:
                                custom_positions = torch.arange(self.kv_schedule.position_embedding_place_examples, self.kv_schedule.position_embedding_place_examples+seq_length).unsqueeze(0).to(self.model.device)
                                batch_custom_positions = custom_positions.repeat(size_B, 1)
                            with torch.no_grad():
                                output = self.model(input_ids=skill_tokens.input_ids,
                                                    attention_mask=concat_attention_mask,
                                                    past_key_values=batch_past_key_values,
                                                    return_dict=True)

                            prompt_last_logits = prompt_output.logits[:, -1:, :].repeat(size_B, 1, 1)  # [B, 1, C]
                            logits = torch.cat((prompt_last_logits, output.logits[:, :-1, :]), dim=1)
                            labels = skill_tokens.input_ids
                            attention_mask = skill_tokens.attention_mask
                    elif self.scoring_mode == 'naive':
                        with torch.no_grad():
                            sentence_tokens = self.tokenizer(batch_sentence, add_special_tokens=False, return_tensors="pt",
                                                             padding=True)
                            sentence_tokens = sentence_tokens.to(self.device)
                            output = self.model(sentence_tokens.input_ids, attention_mask=sentence_tokens.attention_mask,
                                                return_dict=True)
                            logits = output.logits[:, prompt_len - 1:-1]
                            labels = sentence_tokens.input_ids[:, prompt_len:]
                            attention_mask = sentence_tokens.attention_mask[:, prompt_len:]

                    size_B, size_L, size_C = logits.shape
                    logits = logits.reshape([size_B * size_L, size_C])
                    labels = labels.reshape([size_B * size_L])
                    loss_fn = CrossEntropyLoss(reduction='none')
                    loss = loss_fn(logits.float(), labels.long())
                    loss = loss.reshape([size_B, size_L])
                    skill_len = attention_mask.count_nonzero(axis=1)
                    if self.score_function == 'sum':
                        score = -(loss * attention_mask).sum(axis=1)
                    elif self.score_function == 'avg':
                        score = -(loss * attention_mask).sum(axis=1) / skill_len
                    
                    for skill_id, skill in enumerate(batch_skill_set):
                        scores[skill] = score[skill_id].item()
            # retrun the skill name with the highest score
            score = max(scores, key=lambda x: scores[x]).strip()
            # import pdb;pdb.set_trace()
        else:
            assert False, 'unknown scoring mode'
        if self.use_kv:
            return score, final_prompt
        return score

    def plan_whole(self, query):
        step_seq = []
        skill_set_size_seq = []
        # prompt = self.prompt + f'Human: {query}\nRobot: 1.'
        print(f"Input query: {query}")

        prompt_lines = self.prompt.split('\n')
        prompt_examples = prompt_lines[2:]
        example_text = '\n'.join(prompt_examples)
        skills_text = ', '.join([x.strip() for x in self.skill_set])

        self.guidance_program = guidance("""
        {{#system~}}
        You are a robot operating in a home. A human user can ask you to do various tasks and you are supposed to tell the sequence of actions you would do to accomplish your task.
        {{~/system}}
        
        {{#user~}}
        Examples of human instructions and possible your (robot) answers:
        {{example_text}}
        
        Now please answer the sequence of actions for the input instruction.
        You should use one of actions of this list: {{skills_text}}.
        List the actions with comma seperator.
        
        Input user instruction:   
        {{query}}
        {{~/user}}
        
        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=500}}
        {{~/assistant}}
        """)

        # run
        out = self.guidance_program(example_text=example_text, skills_text=skills_text, query=query)
        answer = out['answer']
        print(answer)

        # to list
        answer = answer.replace('Robot: ', '')
        actions = [action.strip(' 1234567890.') for action in answer.split(',')]
        step_seq = actions

        return step_seq, skill_set_size_seq

    def plan_step_by_step(self, query, prev_steps=(), prev_msgs=()):
        if len(prev_steps) >= self.max_steps:
            return None, None
        if self.use_memory and not self.use_kv:
            self.prompt = self.init_prompt(self.cfg)
        if self.use_kv:
            prompt = f'Human: {query.strip()}\nRobot: 1. '
        else:
            prompt = self.prompt + f'Human: {query.strip()}\nRobot: 1. '
        step_num = 1
        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if len(msg) > 0:
                prompt += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                prompt += step + f', {i + 2}. '
            step_num += 1
        print(f"\nPrompt: {prompt}\n")
        if self.use_kv:
            score, prompt = self.score(prompt, self.skill_set,step_num,query.strip())
        else:
            score = self.score(prompt, self.skill_set,step_num,query.strip())


        return score, prompt

    def duplicate_past_key_values(self, past_key_values, batch_size):
        batch_past_key_values = []
        for layer in range(len(past_key_values)):
            batch_past_key_values_layer = []
            for kv in range(len(past_key_values[layer])):
                batch_past_key_values_layer.append(past_key_values[layer][kv].repeat(batch_size, 1, 1, 1))
            batch_past_key_values_layer = tuple(batch_past_key_values_layer)
            batch_past_key_values.append(batch_past_key_values_layer)
        batch_past_key_values = tuple(batch_past_key_values)
        return batch_past_key_values
