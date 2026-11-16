
p="$U/exp/FMD/xs/code/build_provenance_addendum1.py"; s=open(p).read()
s=s.replace('timing=dict(add1=24844995,add2=24845277,add3=24846156),analysis=dict(sam31_keys=24845924,vision_identity=24845930))','timing=dict(add1=24844995,add2_cancelled_in_queue_replaced_by_split=24845277,add3_cancelled_in_queue_replaced_by_split=24846156,split=dict(s1=24846330,s2=24846335,s3=24846340,s4=24846341)),analysis=dict(sam31_keys=24845924,vision_identity=24845930,reverify_5ids=24846250))')
s=s.replace('b16flbase=24846183)','b16flbase=24846183,b17arabic_deepseek_ds=24846196)') if 'b16flbase' in s else s.replace('b16florence=24846183)','b16florence=24846183,b17arabic_deepseek_ds=24846196)')
s=s.replace('"SD3.5 medium/large','"arabic-ocr-deepseek-ocr-2 not load-tested: remote code needs DeepseekV2Model (ImportError on transformers 4.46.3); its 472 vision/projector tensors are identical to deepseek-ocr-2 (vision_identity_addendum1.json)",\n  "SD3.5 medium/large')
open(p,"w").write(s); print(s.count("24846196"),s.count("DeepseekV2Model"))
