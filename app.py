import streamlit as st
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from gtts import gTTS
import tempfile
import os


# 初始化图片描述模型
@st.cache_resource
def load_captioning_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


# 初始化故事生成模型（使用Mistral，效果更好）
@st.cache_resource
def load_story_model():
    model_name = "HuggingFaceH4/zephyr-7b-beta"  # 更流畅的故事生成
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True  # 减少内存使用
    )
    return tokenizer, model


# 图片转描述
def img2text(image):
    caption_model = load_captioning_model()
    result = caption_model(image)
    return result[0]["generated_text"]


# 描述扩展为流畅故事（50-100词，无重复）
def text2story(caption):
    tokenizer, model = load_story_model()

    # 精心设计的提示词，引导模型生成有逻辑的故事
    prompt = f"""<|user|>
Write a short, logical children's story (50-80 words) based on this scene: "{caption}"

The story must:
- Have a clear beginning, middle, and end
- Be appropriate for kids aged 3-10
- NOT repeat any sentence or phrase
- Flow naturally from one idea to the next

Story:<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2,  # 防止重复
        no_repeat_ngram_size=3  # 禁止3-gram重复
    )

    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取assistant部分
    story = story.split("<|assistant|>")[-1].strip()

    # 确保长度适中
    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100])
    elif len(words) < 50:
        story = story + " The end."

    return story


# 备用方案：如果上述模型太慢，使用GPT-2但改进提示词
def text2story_fallback(caption):
    pipe = pipeline("text-generation", model="distilgpt2")

    prompt = f"""Write a short children's story about {caption}. 
    Begin with "Once upon a time". Each sentence must be different.
    The story should have a happy ending.
    Story: Once upon a time,"""

    result = pipe(
        prompt,
        max_length=130,
        temperature=0.8,
        do_sample=True,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2
    )

    story = result[0]["generated_text"].replace(prompt, "").strip()
    if not story.startswith("Once upon a time"):
        story = "Once upon a time, " + story

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

    # 模型选择（让用户根据硬件选择）
    model_option = st.radio(
        "Choose story quality:",
        ["🎯 Best Quality (Zephyr-7B, slower) - Recommended", "⚡ Faster (GPT-2, less quality)"],
        horizontal=True
    )
    use_fallback = "Faster" in model_option

    # 上传图片
    uploaded_image = st.file_uploader("📸 Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Your Picture", use_container_width=True)

        if st.button("✨ Generate Story ✨", type="primary"):
            # Step 1: 图片转描述
            with st.spinner("🔍 Looking at your picture..."):
                caption = img2text(image)
                st.info(f"📷 **What I see:** {caption}")

            # Step 2: 生成故事
            with st.spinner("📝 Writing a magical story (this takes a moment)..."):
                if use_fallback:
                    story = text2story_fallback(caption)
                else:
                    try:
                        story = text2story(caption)
                    except Exception as e:
                        st.warning(f"Using faster mode due to: {e}")
                        story = text2story_fallback(caption)

                st.success("✅ Story ready!")
                st.markdown("### 📖 Your Unique Story")
                st.write(story)

                # 显示字数统计
                word_count = len(story.split())
                st.caption(f"📊 Word count: {word_count} words")

            # Step 3: 转音频
            with st.spinner("🔊 Turning story into audio..."):
                audio_file = text2audio(story)
                st.audio(audio_file, format="audio/mp3")
                os.unlink(audio_file)

            st.balloons()
            st.audio("Play again", audio_file)  # 确保音频可见

    # 使用说明
    with st.expander("📌 Tips for best results"):
        st.markdown("""
        - **Simple images** (one main object/animal) work best
        - **Best Quality mode** needs ~6GB RAM but gives much better stories
        - **Faster mode** works on any computer but may have some repetition
        - Stories are designed to be logical with clear beginning, middle, and end
        """)


if __name__ == "__main__":
    main()