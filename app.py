import streamlit as st
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from gtts import gTTS
import tempfile
import os
from transformers import BitsAndBytesConfig

# 初始化图片描述模型
@st.cache_resource
def load_captioning_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# 初始化故事生成模型（使用更小的模型，适合Streamlit Cloud）
@st.cache_resource
def load_story_model():
    # 使用更小的模型，避免内存不足
    model_name = "microsoft/phi-2"  # 2.7B参数，比Zephyr小很多
    
    # 使用4bit量化配置（新语法）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

# 图片转描述
def img2text(image):
    caption_model = load_captioning_model()
    result = caption_model(image)
    return result[0]["generated_text"]

# 描述扩展为流畅故事
def text2story(caption):
    try:
        tokenizer, model = load_story_model()
        
        prompt = f"""Write a short children's story (50-80 words) about: {caption}
        
The story must:
- Have a clear beginning, middle, and end
- Be fun for kids aged 3-10
- Not repeat any phrases
- Flow naturally

Story:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # 将输入移动到相同设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取故事部分
        story = story.split("Story:")[-1].strip()
        
        # 确保长度
        words = story.split()
        if len(words) > 100:
            story = " ".join(words[:100])
        elif len(words) < 50:
            story = story + " The end."
        
        return story
        
    except Exception as e:
        # 如果phi-2失败，使用备用方案
        st.warning(f"Using fallback story generator: {e}")
        return text2story_fallback(caption)

# 备用方案：使用GPT-2
def text2story_fallback(caption):
    pipe = pipeline("text-generation", model="distilgpt2")
    
    prompt = f"Once upon a time, {caption} Then,"
    
    result = pipe(
        prompt,
        max_length=120,
        temperature=0.8,
        do_sample=True,
        repetition_penalty=1.5,
        pad_token_id=50256,
        truncation=True
    )
    
    story = result[0]["generated_text"]
    # 去除可能重复的部分
    sentences = story.split('. ')
    unique_sentences = []
    for s in sentences:
        if s not in unique_sentences:
            unique_sentences.append(s)
    story = '. '.join(unique_sentences[:4]) + '.'
    
    return story

# 故事转音频
def text2audio(story_text):
    tts = gTTS(text=story_text, lang="en", slow=False)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# 主应用
def main():
    st.set_page_config(page_title="AI Storyteller for Kids", page_icon="📖")
    
    st.title("📖 AI Storyteller for Kids")
    st.write("✨ Upload any picture, and AI will create a unique, logical story just for you!")
    
    # 模型选择
    model_option = st.radio(
        "Choose story quality:",
        ["🎯 Better Quality (Phi-2)", "⚡ Faster (GPT-2)"],
        horizontal=True
    )
    use_fallback = "Faster" in model_option
    
    # 上传图片
    uploaded_image = st.file_uploader("📸 Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        # 修复：使用 width='stretch' 替代 use_container_width
        st.image(image, caption="Your Picture", width='stretch')
        
        if st.button("✨ Generate Story ✨", type="primary"):
            # Step 1: 图片转描述
            with st.spinner("🔍 Looking at your picture..."):
                caption = img2text(image)
                st.info(f"📷 **What I see:** {caption}")
            
            # Step 2: 生成故事
            with st.spinner("📝 Writing a magical story..."):
                if use_fallback:
                    story = text2story_fallback(caption)
                else:
                    story = text2story(caption)
                
                st.success("✅ Story ready!")
                st.markdown("### 📖 Your Unique Story")
                st.write(story)
                
                # 显示字数统计
                word_count = len(story.split())
                st.caption(f"📊 Word count: {word_count} words")
            
            # Step 3: 转音频
            with st.spinner("🔊 Turning story into audio..."):
                audio_file = text2audio(story)
                # 修复：第一个参数应该是音频文件路径
                st.audio(audio_file, format="audio/mp3")
                # 清理临时文件
                os.unlink(audio_file)
            
            st.balloons()
    
    # 使用说明
    with st.expander("📌 Tips for best results"):
        st.markdown("""
        - **Simple images** (one main object/animal) work best
        - **Better Quality mode** gives better stories but takes longer
        - **Faster mode** works quickly but may have some repetition
        - Stories are designed to be logical with clear beginning, middle, and end
        """)

if __name__ == "__main__":
    main()
