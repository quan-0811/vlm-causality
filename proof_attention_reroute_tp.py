#!/usr/bin/env python3
"""BBox-only attention edit for the 10 LLaVA object-reading circuit heads.

At object-last and `in`, move queried-object bbox mass to BOS or uniformly to
outside-bbox image patches. Preserve the actual head output and add only the
intended delta: z_new = z_actual + (A_edited - A_clean) @ V.
"""
import argparse, json, math, re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

HEADS=((11,17),(7,14),(10,29),(11,26),(11,14),(8,6),(8,25),(12,25),(6,17),(11,7))
ALIASES={'people':'person','persons':'person','bike':'bicycle','motorbike':'motorcycle','plane':'airplane','sofa':'couch','tv monitor':'tv','cellphone':'cell phone'}

def args():
 p=argparse.ArgumentParser(); p.add_argument('--question-id',type=int,default=1)
 p.add_argument('--pope-file',default='pope_train/coco_train_pope_adversarial.jsonl')
 p.add_argument('--image-root',default='data/coco2014/train2014/train2014')
 p.add_argument('--instances',default='data/coco2014/captions/annotations/instances_train2014.json')
 p.add_argument('--model-id',default='llava-hf/llava-1.5-7b-hf')
 p.add_argument('--output',default='proof_attention_reroute_tp_result.json'); return p.parse_args()

def load(path):
 text=Path(path).read_text().strip()
 return json.loads(text) if text.startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]

def question(item): return item.get('prompt',item.get('text',item.get('question')))
def object_name(text):
 m=re.search(r'Is there (?:a|an) (.+?) in the image',text,re.I)
 if not m: raise ValueError(f'Cannot parse object: {text}')
 name=m.group(1).strip().lower(); return ALIASES.get(name,name)

def build_inputs(processor,item,image_root):
 image=Image.open(Path(image_root)/item['image']).convert('RGB')
 message=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':question(item)}]}]
 batch=processor.apply_chat_template(message,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
 return image,batch

def semantic_positions(batch,tokenizer,image_id,bos_id):
 ids=batch['input_ids'][0]; tokens=tokenizer.convert_ids_to_tokens(ids.tolist())
 normalized=[x.replace('▁','').lower() for x in tokens]
 ins=[i for i,x in enumerate(normalized) if x=='in']; images=torch.where(ids==image_id)[0]; bos=torch.where(ids==bos_id)[0]
 if not ins or not len(images) or not len(bos): raise RuntimeError('Missing in/image/BOS positions')
 pos=ins[-1]; return (pos-1,pos),images,int(bos[0]),tokens

def annotations(path):
 coco=json.loads(Path(path).read_text()); cats={x['name'].lower():int(x['id']) for x in coco['categories']}; boxes=defaultdict(list)
 for ann in coco['annotations']: boxes[(int(ann['image_id']),int(ann['category_id']))].append(ann['bbox'])
 return cats,boxes

def bbox_patches(item,image,image_positions,processor,cats,all_boxes):
 target=object_name(question(item)); cat=cats.get(target)
 image_id=int(re.search(r'(\d+)\.jpg$',item['image']).group(1)); boxes=all_boxes.get((image_id,cat),[])
 if not boxes: raise RuntimeError(f'No COCO bbox for {target} in {item["image"]}')
 ow,oh=image.size; ch=int(processor.image_processor.crop_size['height']); cw=int(processor.image_processor.crop_size['width'])
 shortest=int(processor.image_processor.size['shortest_edge']); scale=shortest/min(ow,oh)
 rw,rh=round(ow*scale),round(oh*scale); left,top=(rw-cw)/2,(rh-ch)/2; grid=math.isqrt(len(image_positions))
 if grid*grid!=len(image_positions): raise RuntimeError('Non-square image-token grid')
 selected=[]
 for flat in range(grid*grid):
  row,col=divmod(flat,grid); px0,px1=col*cw/grid,(col+1)*cw/grid; py0,py1=row*ch/grid,(row+1)*ch/grid
  for x,y,w,h in boxes:
   bx0,bx1=max(0.,x*scale-left),min(float(cw),(x+w)*scale-left); by0,by1=max(0.,y*scale-top),min(float(ch),(y+h)*scale-top)
   if min(px1,bx1)>max(px0,bx0) and min(py1,by1)>max(py0,by0): selected.append(flat); break
 if not selected: raise RuntimeError('BBox has no image patch after preprocessing')
 return torch.tensor(selected),boxes,target,grid

def rotate_half(x):
 a,b=x.chunk(2,dim=-1); return torch.cat((-b,a),dim=-1)
def rope(x,positions,inv_freq):
 f=torch.outer(positions.float(),inv_freq.to(x.device)); e=torch.cat((f,f),dim=-1)
 return x.float()*e.cos()+rotate_half(x.float())*e.sin()

@contextmanager
def edit(mode,layers,by_layer,queries,bbox,outside,bos,n_heads,n_kv,head_dim,inv_freq,method='delta'):
 repeat=n_heads//n_kv; qcache={}; kcache={}; vcache={}; records={}; hooks=[]
 for layer,heads in by_layer.items():
  attn=layers[layer].self_attn
  def qhook(_m,_a,out,layer=layer): qcache[layer]=out.reshape(out.shape[0],out.shape[1],n_heads,head_dim)
  def khook(_m,_a,out,layer=layer): kcache[layer]=out.reshape(out.shape[0],out.shape[1],n_kv,head_dim)
  def vhook(_m,_a,out,layer=layer): vcache[layer]=out.reshape(out.shape[0],out.shape[1],n_kv,head_dim)
  def ohook(_m,a,layer=layer,heads=tuple(heads)):
   hidden=a[0]; z=hidden.reshape(hidden.shape[0],hidden.shape[1],n_heads,head_dim).clone(); bp=bbox.to(z.device); op=outside.to(z.device)
   for head in heads:
    kv=head//repeat
    for slot,query_pos in enumerate(queries):
     allowed=torch.arange(query_pos+1,device=z.device); qr=qcache[layer][0,query_pos,head]; kr=kcache[layer][0,:query_pos+1,kv]; vals=vcache[layer][0,:query_pos+1,kv].float()
     qrot=rope(qr[None],torch.tensor([query_pos],device=z.device),inv_freq)[0]; krot=rope(kr,allowed,inv_freq)
     weights=(qrot@krot.T/math.sqrt(head_dim)).softmax(-1); edited=weights.clone(); removed=edited[bp].sum(); edited[bp]=0
     if mode=='bbox_to_bos': edited[bos]+=removed
     else: edited[op]+=removed/len(op)
     delta=(edited-weights)@vals; actual=z[0,query_pos,head].float(); reconstructed=weights@vals
     if method=='delta': z[0,query_pos,head]=(actual+delta).to(z.dtype)
     elif method=='direct': z[0,query_pos,head]=(edited@vals).to(z.dtype)
     else: raise ValueError(method)
     records[(layer,head,slot)]={'bbox_mass':float(removed.cpu()),'delta_norm':float(delta.norm().cpu()),'reconstruction_error':float(((reconstructed-actual).norm()/actual.norm().clamp_min(1e-12)).cpu())}
   return (z.reshape_as(hidden),)+a[1:]
  hooks += [attn.q_proj.register_forward_hook(qhook),attn.k_proj.register_forward_hook(khook),attn.v_proj.register_forward_hook(vhook),attn.o_proj.register_forward_pre_hook(ohook)]
 try: yield records
 finally:
  for h in hooks: h.remove()

def stats(logits,yes_id,no_id):
 x=logits[0,-1].float(); y,n=x[yes_id].item(),x[no_id].item(); p=torch.softmax(x[[yes_id,no_id]],0)
 return {'prediction':'yes' if y>n else 'no','yes_logit':y,'no_logit':n,'logit_diff_yes_minus_no':y-n,'binary_p_yes':p[0].item(),'binary_p_no':p[1].item()}

def main():
 a=args(); items={int(x['question_id']):x for x in load(a.pope_file)}; item=items[a.question_id]
 processor=AutoProcessor.from_pretrained(a.model_id,use_fast=False); dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
 model=LlavaForConditionalGeneration.from_pretrained(a.model_id,dtype=dtype,device_map='auto' if torch.cuda.is_available() else None,attn_implementation='eager',low_cpu_mem_usage=True).eval()
 cfg=model.config.text_config; lm=model.language_model if hasattr(model,'language_model') else model.model.language_model; layers=lm.layers if hasattr(lm,'layers') else lm.model.layers; inv_freq=lm.rotary_emb.inv_freq.detach().float()
 image,cpu=build_inputs(processor,item,a.image_root); queries,image_pos,bos,tokens=semantic_positions(cpu,processor.tokenizer,model.config.image_token_index,processor.tokenizer.bos_token_id)
 cats,boxes=annotations(a.instances); bbox_flat,coco_boxes,obj,grid=bbox_patches(item,image,image_pos,processor,cats,boxes); bbox=image_pos[bbox_flat]; mask=torch.ones(len(image_pos),dtype=torch.bool); mask[bbox_flat]=False; outside=image_pos[mask]
 inputs={k:(v.to(model.device).to(dtype) if k=='pixel_values' else v.to(model.device)) for k,v in cpu.items()}; yes=processor.tokenizer.encode(' Yes',add_special_tokens=False)[-1]; no=processor.tokenizer.encode(' No',add_special_tokens=False)[-1]
 by_layer=defaultdict(list)
 for layer,head in HEADS: by_layer[layer].append(head)
 with torch.inference_mode(): baseline=stats(model(**inputs,use_cache=False,return_dict=True).logits,yes,no)
 conditions={'baseline':baseline}; diagnostics={}
 cases=(('bbox_to_bos_delta','bbox_to_bos','delta'),('bbox_to_bos_direct','bbox_to_bos','direct'),('bbox_to_outside_delta','bbox_to_outside','delta'),('bbox_to_outside_direct','bbox_to_outside','direct'))
 for name,mode,method in cases:
  with edit(mode,layers,by_layer,queries,bbox,outside,bos,cfg.num_attention_heads,getattr(cfg,'num_key_value_heads',cfg.num_attention_heads),cfg.hidden_size//cfg.num_attention_heads,inv_freq,method) as rec:
   with torch.inference_mode(): conditions[name]=stats(model(**inputs,use_cache=False,return_dict=True).logits,yes,no)
  diagnostics[name]={'nodes':len(rec),'mean_bbox_mass':sum(x['bbox_mass'] for x in rec.values())/len(rec),'mean_delta_norm':sum(x['delta_norm'] for x in rec.values())/len(rec),'mean_reconstruction_error':sum(x['reconstruction_error'] for x in rec.values())/len(rec)}
 base=baseline['logit_diff_yes_minus_no']; result={'design':'compare actual+delta against direct edited_attention@V','question_id':a.question_id,'image':item['image'],'question':question(item),'object':obj,'heads':[f'L{l}H{h}' for l,h in HEADS],'queries':[{'slot':'object_last','position':queries[0],'token':tokens[queries[0]]},{'slot':'in','position':queries[1],'token':tokens[queries[1]]}],'grid':grid,'bbox_patches':len(bbox_flat),'image_patches':len(image_pos),'coco_boxes':coco_boxes,'dtype':str(dtype),'backend':'eager','conditions':conditions,'delta':{name:conditions[name]['logit_diff_yes_minus_no']-base for name,_,_ in cases},'diagnostics':diagnostics}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
