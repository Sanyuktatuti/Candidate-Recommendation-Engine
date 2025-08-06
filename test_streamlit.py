"""
Simple Streamlit test to verify file upload works.
Run with: streamlit run test_streamlit.py --server.port 8502
"""
import streamlit as st

st.set_page_config(page_title="Test File Upload", layout="wide")

st.title("🧪 File Upload Test")

# Sidebar
with st.sidebar:
    st.header("Test Options")
    method = st.selectbox("Choose Method", ["File Upload", "Text Input"])

st.header("Main Content")

if method == "File Upload":
    st.subheader("📁 File Upload Test")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload any PDF, DOCX, or TXT files"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
        
        for file in uploaded_files:
            st.write(f"• **{file.name}** ({file.size} bytes)")
    else:
        st.info("👆 Please upload some files using the file uploader above")

else:
    st.subheader("✏️ Text Input Test")
    text_input = st.text_area("Enter some text:", height=100)
    
    if text_input:
        st.success("✅ Text input received!")
        st.write(f"Length: {len(text_input)} characters")

st.write("---")
st.info("If you can see this page and the file uploader above, then Streamlit is working correctly!")
