import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import sys 
import time # Used for measuring training time
import altair as alt # Import Altair for charting

# ===============================
# 3️⃣ Feedforward model (Dynamic layers)
# ===============================
class SentimentNet(nn.Module):
    """Simple Feedforward Network for Sentiment Classification with dynamic hidden layers."""
    def __init__(self, input_dim, hidden_dims_list):
        super(SentimentNet, self).__init__()
        
        layer_dims = [input_dim] + hidden_dims_list + [1]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        return torch.tanh(self.layers[-1](x))

# ===============================
# 1️⃣ Setup (Banking Theme & Layout)
# ===============================
st.set_page_config(page_title="Sentinel Chat AI - Sentiment Analyzer", layout="wide")

# --- Sidebar Theming (Mock Banking Menu) ---
st.sidebar.markdown(
    """
    <style>
    .sidebar-logo {
        font-size: 28px;
        font-weight: bold;
        color: #004c99; /* Deep Bank Blue */
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .menu-item {
        padding: 10px 0;
        cursor: pointer;
        font-size: 16px;
        color: #333333;
    }
    .menu-item:hover {
        color: #007bff; /* Brighter Blue for hover */
    }
    /* Custom styling for metrics to align arrow and text */
    .st-dl .st-b5:first-child > div > div:first-child {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    <div class="sidebar-logo">🏦 Sentinel Bank AI</div>
    <div class="menu-item">🏠 Dashboard</div>
    <div class="menu-item">💳 Accounts</div>
    <div class="menu-item">💰 Transactions</div>
    <div class="menu-item">💬 **Customer Service (Active)**</div>
    <div class="menu-item">⚙️ Settings</div>
    <hr>
    """, unsafe_allow_html=True
)


# ===============================
# 2️⃣ Expanded Training Data (for demo)
# ===============================
data = {
    "text": [
        "I love this product, it's perfect and exceeded my expectations!", "This application is absolutely terrible and unusable, a complete waste of time.", 
        "It works fine most of the time, no major issues, just average performance.", "I’m so disappointed with the lack of features and constant bugs.", 
        "Absolutely fantastic experience, top-notch support and incredibly quick resolution!", "Horrible service! I waited over an hour for a response and got no help.", 
        "Not bad at all, could be better but it serves its basic purpose well.", "Worst thing ever, I'm canceling my subscription right now, I'm furious.",
        "Great help from support, they were very prompt, efficient, and friendly.", "Okay I guess, nothing special about it, quite neutral actually.",
        "The interface is confusing and clunky, a nightmare to navigate.", "The speed is impressive, it loads instantly, highly recommend.",
        "I hate the new update, everything is broken and difficult to use.", "Everything is running smoothly since the last patch, thank thank you for fixing it.",
        "This is an outrage, I demand you fix it immediately or I'm escalating.", "I feel much better after talking to your agent, they were very soothing and helpful.",
        "I'm satisfied with the current state of my account.", "My problem wasn't solved, which is annoying.",
        "The customer service representative was rude.", "I appreciate the effort your team put in.",
        "The service was bad and my issue wasn't fixed.",
        "I have no opinion on the matter."
    ],
    "label": [0.95, -0.9, 0.3, -0.85, 1.0, -1.0, 0.4, -1.0, 0.85, 0.0, -0.7, 0.9, -0.95, 0.7, -0.8, 0.6, 0.5, -0.5, -0.9, 0.7, -0.8, 0.0] 
}
df = pd.DataFrame(data)

# ===============================
# 4️⃣ Session State Initialization (Dynamic layers)
# ===============================
if 'hidden_dims_str' not in st.session_state:
    st.session_state.hidden_dims_str = "128, 64" # Default to two layers
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cpu")
if 'sim_delay' not in st.session_state:
    st.session_state.sim_delay = 1.0
if 'sim_index' not in st.session_state:
    st.session_state.sim_index = 0
if 'train_epochs' not in st.session_state:
    st.session_state.train_epochs = 100
if 'current_convo_data' not in st.session_state:
    st.session_state.current_convo_data = [] # Stores the currently running conversation messages

def get_hidden_dims(dims_str):
    """Parses comma-separated string into a list of valid integer dimensions."""
    try:
        dims = [int(d.strip()) for d in dims_str.split(',') if d.strip().isdigit() and int(d.strip()) > 0]
        return dims
    except Exception:
        return [128, 64] 

if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = TfidfVectorizer(max_features=500)
    X = st.session_state.vectorizer.fit_transform(df["text"]).toarray()
    y = df["label"].values
    
    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_dim = st.session_state.X_train.shape[1]
    hidden_dims = get_hidden_dims(st.session_state.hidden_dims_str)
    
    st.session_state.model = SentimentNet(
        input_dim, 
        hidden_dims
    ) 

if "messages" not in st.session_state:
    st.session_state.messages = []
if "scores" not in st.session_state:
    # scores tracks aggregate sentiment for 'agent' and 'customer' roles
    st.session_state.scores = {"agent": [], "customer": []}
if "simulation_active" not in st.session_state:
    st.session_state.simulation_active = False

# Retrieve model and data from state
model = st.session_state.model
vectorizer = st.session_state.vectorizer
X_train = st.session_state.X_train
y_train = st.session_state.y_train
device = st.session_state.device 

# ===============================
# 5️⃣ Core Analysis Function (UPDATED for Agent Name)
# ===============================
def analyze_and_send(msg, sender_full_name):
    """Analyzes sentiment and updates session state."""
    
    # Determine the underlying role ('Customer' or 'Agent') for scoring aggregates
    sender_type = "Agent" if sender_full_name.startswith("Agent") else "Customer"
    score_key = sender_type.lower()
    
    if not msg or score_key not in ["agent", "customer"]:
        return
    
    if 'model' not in st.session_state:
        st.error("Model not initialized. Please configure layers and click 'Train Model'.")
        return

    try:
        model = st.session_state.model
        model.eval()
        model.to(device)
        
        vec = vectorizer.transform([msg]).toarray()
        vec_tensor = torch.FloatTensor(vec).to(device)
        
        with torch.no_grad():
            score = model(vec_tensor).item()
            
        # Store the full name for display
        st.session_state.messages.append({"sender": sender_full_name, "content": msg, "score": score})
        # Store score based on role type
        st.session_state.scores[score_key].append(score)
    except Exception as e:
        print(f"Error during sentiment analysis. Please try re-training the model: {e}", file=sys.stderr)
        st.error("Sentiment analysis failed. Please ensure the model is trained with valid layer dimensions.")


# ===============================
# 6️⃣ Simulation Logic and Data (UPDATED: Multiple Threads)
# ===============================
SIMULATED_CONVERSATIONS = {
    "Agent Alice - Order Delay Resolution (Positive)": [
        ("Customer", "I'm calling about order #4829, it hasn't arrived. I'm very upset."),
        ("Agent Alice", "I apologize for the delay. Let me check the tracking details for you right now."),
        ("Customer", "I waited a week! This is absolutely infuriating, your service is terrible. I demand a refund."),
        ("Agent Alice", "I understand your frustration. It looks like the package was delayed due to a regional weather issue, which is out of our control."),
        ("Agent Alice", "However, the good news is, it's now out for delivery and should arrive within the next 2 hours."),
        ("Customer", "Seriously? Why wasn't I notified? That's still poor communication, even with the weather delay."),
        ("Agent Alice", "You are correct, and I apologize for the lack of communication. I'm adding a $15 credit to your account for the inconvenience."),
        ("Customer", "A credit? Wow, that's actually quite generous. Thank you for resolving this for me so quickly."),
        ("Agent Alice", "Is there anything else I can help you with today?"),
    ],
    "Agent Bob - Technical Issue (Negative)": [
        ("Customer", "My mobile app keeps crashing every time I try to deposit a check. It's version 3.2."),
        ("Agent Bob", "Thank you for that information. Can you confirm your phone operating system version?"),
        ("Customer", "It's the latest iOS, 17.1. I've tried reinstalling the app twice, it doesn't work. This is making me miss deadlines!"),
        ("Agent Bob", "I see. It appears there is a known compatibility bug with iOS 17.1 that our development team is working on."),
        ("Agent Bob", "The current workaround is to use the web browser portal for check deposits."),
        ("Customer", "A workaround? That's not acceptable! I need the mobile app to work. Why are you rolling out broken software?"),
        ("Agent Bob", "I understand your disappointment. We do not have an ETA for the fix, but we assure you it's prioritized."),
        ("Customer", "This is the worst technical support I've ever received. My issue is completely unresolved. Goodbye."),
    ],
    "Agent Carol - Account Update (Neutral/Procedural)": [
        ("Customer", "I need to update my mailing address from 123 Main St to 456 Elm Ave."),
        ("Agent Carol", "Certainly. I can make that change for you. Please confirm your date of birth for verification."),
        ("Customer", "10/15/1985."),
        ("Agent Carol", "Thank you. I have successfully updated your address in our system to 456 Elm Ave. You should receive a confirmation email shortly."),
        ("Customer", "Great, that was straightforward. Thanks."),
        ("Agent Carol", "You're welcome. Is there anything else I can assist you with regarding your account?"),
    ],
}
if 'selected_convo_key' not in st.session_state:
    st.session_state.selected_convo_key = list(SIMULATED_CONVERSATIONS.keys())[0]


# ----------------------------------------------------
# Helper Function for Color-Coded Arrows
# ----------------------------------------------------
def get_sentiment_arrow_html(score, include_score=False):
    """
    Generates an HTML string for a color-coded arrow and score based on score (-1 to 1).
    Interpolates color from Red (Negative -1) through White/Gray (Neutral 0) to Green (Positive 1).
    """
    
    # Clamp score to the -1.0 to 1.0 range
    score = max(-1.0, min(1.0, score))
    
    # Determine direction (Arrow character)
    if score > 0.05: 
        arrow = "▲"
    elif score < -0.05: 
        arrow = "▼"
    else:
        arrow = "●"

    # Color Interpolation: R, G, B range from 0 to 255
    if score > 0:
        abs_score = score
        r = int(200 * (1 - abs_score))
        g = int(150 + (255 - 150) * abs_score)
        b = int(200 * (1 - abs_score))
        
    elif score < 0:
        abs_score = -score
        r = int(150 + (255 - 150) * abs_score)
        g = int(200 * (1 - abs_score))
        b = int(200 * (1 - abs_score))
        
    else: # Near zero is pure light gray/white
        r, g, b = 200, 200, 200

    color_hex = f'#{r:02x}{g:02x}{b:02x}'
    
    # Build the HTML output
    score_text = f" ({score:.2f})" if include_score else ""
    html = f'<span style="color: {color_hex}; font-weight: bold; font-size: 18px; margin-right: 5px;">{arrow}</span>{score_text}'
    return html

# ==========================================================
# 7️⃣ Main App Content (Single Page Layout)
# ==========================================================

# ----------------------------------------------------
# Training Function Definition (Includes Timing Logic)
# ----------------------------------------------------
def run_training():
    
    def format_time(seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}m {seconds:02d}s"

    # 1. Parse the dynamic hidden dimensions from the user input
    try:
        hidden_dims = get_hidden_dims(st.session_state.hidden_dims_str)
        if not hidden_dims:
            st.error("Please enter at least one valid, positive integer for the hidden layer dimensions.")
            return
    except Exception:
        st.error("Invalid layer dimensions format. Please use comma-separated integers (e.g., 128, 64).")
        return

    # 2. Re-initialize model with new dynamic parameters
    input_dim = st.session_state.X_train.shape[1]
    st.session_state.model = SentimentNet(
        input_dim, 
        hidden_dims
    )
    
    # 3. Training setup
    model = st.session_state.model
    model.to(st.session_state.device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    X_train_tensor = torch.FloatTensor(st.session_state.X_train).to(st.session_state.device)
    y_train_tensor = torch.FloatTensor(st.session_state.y_train).unsqueeze(1).to(st.session_state.device)
    
    # 4. Start timing and training
    start_time = time.time()
    first_epoch_duration = 0.0
    
    for epoch in range(st.session_state.train_epochs):
        epoch_start_time = time.time()
        
        model.train()
        optimizer.zero_grad()
        out = model(X_train_tensor)
        loss = criterion(out, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # --- TIME CALCULATION ---
        epoch_end_time = time.time()
        current_epoch_duration = epoch_end_time - epoch_start_time
        
        if epoch == 0:
            # Base the ETR calculation on the first epoch duration
            first_epoch_duration = current_epoch_duration
        
        epochs_completed = epoch + 1
        epochs_remaining = st.session_state.train_epochs - epochs_completed
        
        # ETR calculation
        time_remaining = first_epoch_duration * epochs_remaining
        total_elapsed_time = epoch_end_time - start_time

        # --- UPDATE PROGRESS AND TIME DISPLAY ---
        percent_completed = epochs_completed / st.session_state.train_epochs
        
        training_progress_placeholder.progress(
            percent_completed, 
            text=f"Epoch {epochs_completed}/{st.session_state.train_epochs} running. Loss: {loss.item():.4f}"
        )
        
        # Show elapsed and estimated time remaining
        training_time_placeholder.markdown(
            f"""
            **Time:** Elapsed `{format_time(total_elapsed_time)}` | 
            ETR `{format_time(time_remaining)}`
            """
        )
    
    model.eval()
    
    architecture_summary = " → ".join(map(str, hidden_dims))
        
    training_progress_placeholder.empty() # Clear the progress bar after training
    training_time_placeholder.empty() # Clear time metrics after success
    st.success(f"✅ Training completed on **{st.session_state.device}** with architecture: **{input_dim} → {architecture_summary} → 1**")


# ----------------------------------------------------
# NEW Section 1: ML Configuration and Training
# ----------------------------------------------------
st.header("⚙️ ML Configuration and Training")
st.markdown("Configure the neural network and training environment. Click **Train Model** to re-train the sentiment analysis logic.")

# --- Main Columns (Device on Left, Training/Model on Right) ---
col_device_group, col_train_config = st.columns([1, 2])

# --- 1. Left Column: Compute Device ---
with col_device_group:
    st.subheader("Compute Device")
    num_gpus = torch.cuda.device_count()
    gpu_options = ["CPU"] + [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(num_gpus)]

    def update_device():
        selected = st.session_state.device_select
        if selected == "CPU" or num_gpus == 0:
            st.session_state.device = torch.device("cpu")
        else:
            try:
                device_index = int(selected.split()[1].replace(":", ""))
                st.session_state.device = torch.device(f"cuda:{device_index}")
            except (IndexError, ValueError):
                st.session_state.device = torch.device("cpu")
        st.info(f"Analysis and training will now use: **{st.session_state.device}**")

    st.selectbox(
        "Select training/inference device", 
        gpu_options,
        index=gpu_options.index(str(st.session_state.device).upper()) if str(st.session_state.device).upper() in gpu_options else 0,
        key='device_select',
        on_change=update_device,
        disabled=st.session_state.simulation_active
    )
    st.markdown(f"**Currently set:** `{st.session_state.device}`")

# --- 2. Right Column: Training Params and Architecture (Combined) ---
with col_train_config:
    st.subheader("Training Parameters & Architecture")
    
    col_epochs, col_architecture = st.columns(2)
    
    with col_epochs:
        st.session_state.train_epochs = st.slider(
            "Training epochs", 
            10, 300, 100, 10, 
            key='train_epochs_slider',
            disabled=st.session_state.simulation_active
        )
        
    with col_architecture:
        hidden_dims_input = st.text_input(
            "Hidden Layer Dimensions (e.g., 128, 64)",
            value=st.session_state.hidden_dims_str,
            key='config_hidden_dims_str',
            placeholder="e.g., 128, 64, 32",
            disabled=st.session_state.simulation_active
        )
        st.session_state.hidden_dims_str = hidden_dims_input
        
        current_dims = get_hidden_dims(st.session_state.hidden_dims_str)
        if current_dims:
            input_size = st.session_state.X_train.shape[1] if 'X_train' in st.session_state else 'N/A'
            st.markdown(f"**Architecture:** `... → {' → '.join(map(str, current_dims))} → 1`")
        else:
            st.warning("Define valid dimensions.")

    # Placeholder for the progress bar (Below parameter inputs)
    training_progress_placeholder = st.empty()
    # Placeholder for the time metrics (New element)
    training_time_placeholder = st.empty()

    # Train button (Moved to the bottom)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # Small spacer
    st.button("🧠 Train Model", on_click=run_training, use_container_width=True, disabled=st.session_state.simulation_active)

st.divider()


# ----------------------------------------------------
# Live Chat & Batch Upload
# ----------------------------------------------------

st.header("💬 Live Chat Monitoring & Sentiment Analysis")
st.markdown("---")
st.write("Analyze sentiment from **-1.0 (Negative)** to **+1.0 (Positive)**.")

# --- CSV Upload Option (Made compact) ---
st.subheader("📁 Upload Conversation for Batch Analysis")

# --- Added CSV Format Preview ---
with st.expander("❓ View required CSV format"):
    st.markdown("""
    Your CSV file must contain two columns: `sender` and `text`.
    
    The `sender` field for Agents should ideally include the Agent's name (e.g., `Agent Alice` or `Agent Bob`). The field for the customer should be `Customer`.
    
    | sender | text |
    | :--- | :--- |
    | Customer | I cannot log into my account, please help! |
    | Agent Bob | I apologize for the trouble. I can certainly assist you. |
    """)

# Use columns for the uploader and the eventual batch analysis button
col_uploader, col_batch_btn_spacer = st.columns([2, 3])

with col_uploader:
    uploaded_file = st.file_uploader(
        "Choose a CSV file (must have 'sender' and 'text' columns)", 
        type="csv",
        disabled=st.session_state.simulation_active
    )

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        if 'sender' in batch_df.columns and 'text' in batch_df.columns:
            st.success(f"Loaded {len(batch_df)} messages. Click below to analyze.")
            with col_uploader:
                if st.button("Start Batch Analysis", key='batch_start', disabled=st.session_state.simulation_active):
                    with st.spinner('Analyzing batch messages...'):
                        for index, row in batch_df.iterrows():
                            # Use the full sender name from the CSV
                            sender_full_name = str(row['sender'])
                            analyze_and_send(str(row['text']), sender_full_name)
                    st.info("Batch analysis complete. Results added to the Conversation Log.")
                    st.rerun()
        else:
            st.error("CSV must contain columns named 'sender' and 'text'.")
    except Exception as e:
        st.error(f"Failed to read or process CSV file: {e}")

st.markdown("---")


# --- Conversation Log Display ---
chat_container = st.container(height=300, border=True)

with chat_container:
    st.markdown("###### 🗨️ Conversation Log")
    if not st.session_state.messages and not st.session_state.simulation_active:
        st.info("Start a conversation below, upload a batch, or run a full simulation from the sidebar.")

    for message in st.session_state.messages:
        sender_full_name = message["sender"]
        content = message["content"]
        score = message["score"]
        
        # Determine the role (Customer or Agent) for styling
        if sender_full_name.startswith("Agent"):
            role = "Agent"
            display_name = sender_full_name # e.g., "Agent Alice"
            avatar = "🧑‍💼"
        else:
            role = "Customer"
            display_name = "Customer"
            avatar = "👩"
        
        # Get the new HTML-styled arrow including the score
        sentiment_html = get_sentiment_arrow_html(score, include_score=True)

        # Use the base role for chat_message styling, but display the full name
        with st.chat_message(name=role, avatar=avatar):
            st.markdown(f'**{display_name}** | **Sentiment:** {sentiment_html}', unsafe_allow_html=True)
            st.markdown(content)

# --- Manual Chat Input (UPDATED for Agent Name) ---
st.divider()

col_select, col_input = st.columns([1, 4])

with col_select:
    sender_role = st.radio(
        "Sender Type", 
        ("Customer", "Agent"), 
        key="sender_type",
        horizontal=False,
        disabled=st.session_state.simulation_active
    )
    
    # Define the sender name based on the selected role for manual input
    sender_name_for_chat = "Agent Manual" if sender_role == "Agent" else "Customer"

with col_input:
    prompt = st.chat_input(
        f"Send message as {sender_name_for_chat}...",
        disabled=st.session_state.simulation_active 
    )
    if prompt:
        analyze_and_send(prompt, sender_name_for_chat)
        st.rerun() 

# ========================================================
# 9️⃣ Simulation Start Logic (UPDATED: Selectbox Integration)
# ========================================================

def start_simulation():
    """Initializes the simulation state, loads the selected conversation, and triggers the first rerun."""
    # Ensure all states are clean before starting
    st.session_state.simulation_active = True
    st.session_state.messages = []
    st.session_state.scores = {"agent": [], "customer": []}
    st.session_state.sim_index = 0
    
    # Load the selected conversation data
    st.session_state.current_convo_data = SIMULATED_CONVERSATIONS[st.session_state.selected_convo_key_select]
    
    # Explicitly call rerun immediately to ensure clean start
    st.rerun()

# --- Simulation Controls (Keep in sidebar) ---
st.sidebar.subheader("🎬 Simulation Threads")

# Dropdown to select conversation thread
st.session_state.selected_convo_key = st.sidebar.selectbox(
    "Select Conversation Thread",
    list(SIMULATED_CONVERSATIONS.keys()),
    key='selected_convo_key_select',
    disabled=st.session_state.simulation_active
)

st.session_state.sim_delay = st.sidebar.number_input(
    "Delay between messages (seconds)",
    min_value=0.1, max_value=5.0, value=st.session_state.sim_delay, step=0.1,
    key='sim_delay_input',
    disabled=st.session_state.simulation_active
)

# Use the start_simulation function with on_click
st.sidebar.button(
    f"▶️ Start Simulation: {st.session_state.selected_convo_key.split(' - ')[0]}", 
    on_click=start_simulation, # Use on_click for clean state transition
    disabled=st.session_state.simulation_active
)

if st.session_state.simulation_active:
    conversation_data = st.session_state.current_convo_data
    total_messages = len(conversation_data)
    current_index = st.session_state.sim_index
    
    if current_index < total_messages:
        sender, msg = conversation_data[current_index]
        
        # The spinner ensures the UI shows "loading" during the sleep duration
        with st.spinner(f"Processing message {current_index + 1}/{total_messages} ({sender}) with {st.session_state.sim_delay:.1f}s delay..."):
            analyze_and_send(msg, sender)
            time.sleep(st.session_state.sim_delay)
        
        st.session_state.sim_index += 1
        st.rerun()
        
    else:
        st.session_state.simulation_active = False
        st.session_state.sim_index = 0
        st.info(f"Simulation of thread '{st.session_state.selected_convo_key}' completed and recorded to the chat log.")
        st.rerun()


st.sidebar.divider()

# ----------------------------------------------------
# Sentiment Dashboard
# ----------------------------------------------------
def cumulative_avg(scores):
    return np.mean(scores) if scores else 0.0

cust_avg = cumulative_avg(st.session_state.scores["customer"])
agent_avg = cumulative_avg(st.session_state.scores["agent"])
overall_avg = cumulative_avg(
    st.session_state.scores["customer"] + st.session_state.scores["agent"]
)

st.divider()
st.subheader("📈 Real-Time Sentiment Dashboard")

col_metrics = st.columns(3)

# Display Metrics with color-coded arrows
with col_metrics[0]:
    metric_label = "Customer Average Sentiment"
    metric_value = f'{get_sentiment_arrow_html(cust_avg)}{cust_avg:.2f}'
    st.markdown(f"**{metric_label}**")
    st.markdown(metric_value, unsafe_allow_html=True)

with col_metrics[1]:
    metric_label = "Agent Average Sentiment"
    metric_value = f'{get_sentiment_arrow_html(agent_avg)}{agent_avg:.2f}'
    st.markdown(f"**{metric_label}**")
    st.markdown(metric_value, unsafe_allow_html=True)
    
with col_metrics[2]:
    metric_label = "Overall Average Sentiment"
    metric_value = f'{get_sentiment_arrow_html(overall_avg)}{overall_avg:.2f}'
    st.markdown(f"**{metric_label}**")
    st.markdown(metric_value, unsafe_allow_html=True)


# Prepare chart data for plotting score trend
st.markdown("---")
st.markdown("##### Sentiment Trend Over Conversation")
st.markdown("""
The **Y-axis** represents the sentiment score, ranging from **-1.0 (Strong Negative)** at the bottom to **+1.0 (Strong Positive)** at the top. The **0.0** line represents a neutral sentiment.
""")


all_messages = st.session_state.messages
if all_messages:
    
    # --- Data Preparation for Chart 1 (Per-Message) ---
    chart_df_per_message = pd.DataFrame([
        {
            "Message Index": i + 1,
            # Use the underlying role ('Agent' or 'Customer') for chart coloring
            "Sender": "Agent" if m['sender'].startswith("Agent") else "Customer", 
            "Sentiment Score": m['score']
        } for i, m in enumerate(all_messages)
    ])
    
    # --- Data Preparation for Chart 2 (Cumulative Average) ---
    cumulative_data = []
    cust_sum, cust_count = 0, 0
    agent_sum, agent_count = 0, 0
    
    for i, m in enumerate(all_messages):
        score = m['score']
        
        # Update running sums and counts based on role
        if m['sender'].startswith('Customer'):
            cust_sum += score
            cust_count += 1
        elif m['sender'].startswith('Agent'):
            agent_sum += score
            agent_count += 1
            
        # Calculate current running averages, using NaN if no scores yet
        current_cust_avg = cust_sum / cust_count if cust_count > 0 else np.nan
        current_agent_avg = agent_sum / agent_count if agent_count > 0 else np.nan

        cumulative_data.append({
            "Message Index": i + 1,
            "Customer Cumulative Average": current_cust_avg,
            "Agent Cumulative Average": current_agent_avg
        })
    
    chart_df_cumulative = pd.DataFrame(cumulative_data)
    
    # Convert to long format for Altair plotting
    chart_df_cumulative_long = chart_df_cumulative.melt(
        id_vars=['Message Index'],
        value_vars=['Customer Cumulative Average', 'Agent Cumulative Average'],
        var_name='Sender', # Use 'Sender' for color encoding consistency
        value_name='Sentiment Score'
    ).dropna(subset=['Sentiment Score'])


    # --- Chart Layout and Generation ---
    chart_1_col, chart_2_col = st.columns(2)
    
    with chart_1_col:
        st.markdown("##### 1. Sentiment Trend (Per Message)")
        st.markdown("""
        Shows the **real-time score** for each individual message.
        """)
        
        chart_per_message = alt.Chart(chart_df_per_message).mark_line(point=True).encode(
            x=alt.X('Message Index', axis=alt.Axis(tickMinStep=1)),
            y=alt.Y('Sentiment Score', scale=alt.Scale(domain=[-1.0, 1.0])),
            color='Sender',
            tooltip=['Message Index', 'Sender', 'Sentiment Score']
        ).properties(
            title=""
        ).interactive()

        st.altair_chart(chart_per_message, use_container_width=True)
    
    
    with chart_2_col:
        st.markdown("##### 2. Cumulative Average Trend")
        st.markdown("""
        Shows the **running average score** since the beginning of the chat.
        """)
        
        chart_cumulative_avg = alt.Chart(chart_df_cumulative_long).mark_line(point=True).encode(
            x=alt.X('Message Index', axis=alt.Axis(tickMinStep=1)),
            y=alt.Y('Sentiment Score', scale=alt.Scale(domain=[-1.0, 1.0]), title='Cumulative Sentiment Score'),
            color='Sender',
            tooltip=['Message Index', 'Sender', 'Sentiment Score']
        ).properties(
            title=""
        ).interactive()

        st.altair_chart(chart_cumulative_avg, use_container_width=True)

st.markdown("---")

# ===============================
# 🔟 Reset
# ===============================
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Conversation & Scores"):
    st.session_state.messages = []
    st.session_state.scores = {"agent": [], "customer": []}
    st.session_state.simulation_active = False
    st.session_state.sim_index = 0
    st.session_state.current_convo_data = []
    st.rerun()