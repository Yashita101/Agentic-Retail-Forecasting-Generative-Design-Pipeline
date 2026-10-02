import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="MerchMix AI", layout="wide")

# --- H&M Brand CSS Injection ---
st.markdown("""
<style>
    .stApp { background-color: #FBF9F7; color: #222222; font-family: 'Helvetica Neue', sans-serif; }
    h1, h2, h3, h4 { color: #CC071E !important; font-weight: bold; text-transform: uppercase; margin-bottom: 5px; margin-top: 15px;}
    
    /* Force text input labels to be dark and visible */
    [data-testid="stWidgetLabel"] p { color: #222222 !important; font-weight: bold; font-size: 14px;}
    
    /* Fix alert box visibility: target all internal elements to ensure black text */
    div[data-testid="stAlert"] {
        background-color: #f0f0f0 !important;
        border-left: 4px solid #CC071E !important;
    }
    div[data-testid="stAlert"] * {
        color: #222222 !important;
        font-weight: 500;
    }
    
    .instruction-text {
        color: #222222 !important;
        font-size: 13px;
        font-weight: bold;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    
    .stButton>button {
        background-color: #222222 !important; color: #FFFFFF !important;
        border-radius: 0px !important; border: none; padding: 10px 20px; font-weight: bold; width: 100%;
    }
    .stButton>button:hover { background-color: #CC071E !important; }
    
    .metric-card {
        background: white; padding: 20px; border-left: 5px solid #CC071E; 
        box-shadow: 0px 4px 6px rgba(0,0,0,0.05); margin-bottom: 15px; height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header with Logo ---
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.markdown("<h1>H&M Predictive Merchandising</h1>", unsafe_allow_html=True)
    st.markdown("Forecasted volume anchors and autonomous design prototypes for the upcoming season.")
with header_col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/53/H%26M-Logo.svg", width=60)

st.write("") 

# --- Fetch Top Styles ---
try:
    with st.spinner('Loading XGBoost predictions from API...'):
        response = requests.get(f"{API_BASE_URL}/styles/top?limit=3")
        response.raise_for_status()
        top_styles = response.json()["styles"]
except Exception as e:
    st.error("Backend API is offline. Start the FastAPI server on port 8000.")
    st.stop()

# --- Render Ranked Grid ---
cols = st.columns(len(top_styles))
for idx, style in enumerate(top_styles):
    with cols[idx]:
        st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>Rank {idx+1}: {style['category']}</h3>", unsafe_allow_html=True)
        
        orig_url = f"{API_BASE_URL}/styles/{style['style_id']}/image/original"
        img_req = requests.get(orig_url)
        if img_req.status_code == 200:
            st.markdown(f'''
                <div style="height: 300px; display: flex; justify-content: center; align-items: center; margin-bottom: 15px; background: #f9f9f9;">
                    <img src="{orig_url}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.warning("Image missing")
            
        st.markdown(f"<h2>{style['predicted_volume']} units</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 14px; margin-bottom: 0px;'><strong>Score:</strong> {style['prediction_score']:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 12px; color: gray;'>ID: {style['style_id'].split('-')[0]}</p></div>", unsafe_allow_html=True)
        
        if st.button(f"Open Detail View", key=f"btn_{style['style_id']}"):
            st.session_state.selected_style = style['style_id']

# --- Render Detailed View ---
if 'selected_style' in st.session_state and st.session_state.selected_style is not None:
    st.divider()
    
    if st.button("← Close Detail View", key="close_details"):
        st.session_state.selected_style = None
        st.rerun()
    
    detail_res = requests.get(f"{API_BASE_URL}/styles/{st.session_state.selected_style}")
    if detail_res.status_code == 200:
        details = detail_res.json()
        
        prod_info = details['product_information']
        hist_perf = details['historical_performance']
        model_exp = details['model_explanation']
        
        st.markdown(f"<h3>Concept Analysis: {prod_info['category']}</h3>", unsafe_allow_html=True)
        
        img_col1, img_col2, text_col = st.columns([1, 1, 1.5])
        
        with img_col1:
            st.markdown("**Historical Baseline**")
            orig_url = f"{API_BASE_URL}/styles/{details['style_id']}/image/original"
            if requests.get(orig_url).status_code == 200:
                st.markdown(f'''
                    <div style="height: 450px; display: flex; justify-content: center; align-items: center; background-color: #f4f4f4;">
                        <img src="{orig_url}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.error("Original image not found.")
                
        with img_col2:
            st.markdown("**AI Generated Concept**")
            concept_url = f"{API_BASE_URL}/styles/{details['style_id']}/image/concept"
            if requests.get(concept_url).status_code == 200:
                st.markdown(f'''
                    <div style="height: 450px; display: flex; justify-content: center; align-items: center; background-color: #f4f4f4;">
                        <img src="{concept_url}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.error("Concept image not found.")

        with text_col:
            st.markdown(f"**Attributes:** {prod_info['demographic']} | {prod_info['color']} | {prod_info['pattern']}")
            st.markdown(f"**Prediction Score:** {details['prediction_score']:.2f} ({model_exp['confidence_logic']})")
            st.markdown(f"**Target Volume (30d):** {details['predicted_volume']} units")
            st.markdown(f"**Past Volume (30d):** {hist_perf['sales_last_30d']} units ({hist_perf['sales_velocity_status']})")
            
            st.info(f"**Model Reasoning:**\n{model_exp['reasoning']}\n\n*Primary Feature Driver:* {model_exp['primary_driver']}")
            
            st.markdown("<h4>Submit Feedback</h4>", unsafe_allow_html=True)
            user_comment = st.text_input("Optional Comments", key=f"comment_{details['style_id']}", placeholder="e.g., Modify pocket layout...")
            
            # Formatted instruction text
            st.markdown("<p class='instruction-text'>Click an Approve or Reject button below to save this comment.</p>", unsafe_allow_html=True)
            
            f1, f2 = st.columns(2)
            with f1:
                if st.button("Approve (Like)", key=f"like_{details['style_id']}"):
                    payload = {"style_id": details['style_id'], "feedback_type": "like", "comments": user_comment}
                    requests.post(f"{API_BASE_URL}/styles/feedback", json=payload)
                    st.success("Approval logged successfully.")
            with f2:
                 if st.button("Reject (Dislike)", key=f"dislike_{details['style_id']}"):
                    payload = {"style_id": details['style_id'], "feedback_type": "dislike", "comments": user_comment}
                    requests.post(f"{API_BASE_URL}/styles/feedback", json=payload)
                    st.warning("Rejection logged successfully.")
    else:
        st.error("Error fetching style details from API.")

# --- Render Combined Presentation Board ---
st.divider()
st.markdown("<h3>Seasonal Assortment</h3>", unsafe_allow_html=True)

is_board_open = st.session_state.get('show_board', False)
btn_text = "Close Final Concept Board" if is_board_open else "View Final Concept Board"

if st.button(btn_text):
    st.session_state.show_board = not is_board_open
    st.rerun() 

if st.session_state.get('show_board', False):
    board_url = f"{API_BASE_URL}/presentation/board"
    if requests.get(board_url).status_code == 200:
        st.image(board_url, use_column_width=True)
    else:
        st.info("Combined presentation image not found.")