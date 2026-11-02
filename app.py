import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Configuração da página
st.set_page_config(
    page_title="Classificador de Eucalipto - Green Data Science",
    page_icon="🌿",
    layout="centered"
)

# Classes oficiais (por ordem alfabética do ImageFolder)
CLASSES = ['B', 'K', 'N', 'não nutricional', 'normal', 'P']
NUM_CLASSES = len(CLASSES)

# Pré-processamento igual ao do treino
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(model_name, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )
    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
    
    # Carregar os pesos treinados
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        st.error(f"Erro ao carregar os pesos de {weights_path}: {e}")
        return None

    model.to(device)
    model.eval()
    return model

# --- Interface Gráfica (Streamlit) ---
st.title("🌿 Classificador de Deficiências Nutricionais em Eucalipto")
st.markdown("Carrega uma foto de uma folha de *Eucalyptus globulus* para prever o estado nutricional utilizando Deep Learning.")

# Sidebar para escolha do modelo
st.sidebar.header("⚙️ Configurações")
modelo_escolhido = st.sidebar.selectbox(
    "Escolhe o Modelo:",
    ("ResNet50", "EfficientNet-B0")
)

# Mapeamento do ficheiro de pesos
pesos_path = 'resnet50_eucalipto.pth' if modelo_escolhido == "ResNet50" else 'efficientnet_eucalipto.pth'

# Carregar o modelo selecionado
model = load_model(modelo_escolhido, pesos_path)

# Upload de imagem
uploaded_file = st.file_uploader("Escolhe uma imagem de folha (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagem carregada
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Folha carregada para análise', use_column_width=True)
    
    if model is not None:
        if st.button('🔍 Prever Estado Nutricional'):
            with st.spinner(f'A analisar com {modelo_escolhido}...'):
                # Processar imagem
                input_tensor = transform(image).unsqueeze(0) # Adicionar dimensão de batch
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Inferência
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
                
                # Obter classe prevista e confiança
                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_class = CLASSES[predicted_idx.item()]
                
                # Resultados
                st.success(f"### Resultado da Predição: **{predicted_class}**")
                st.metric(label="Nível de Confiança", value=f"{confidence.item() * 100:.2f}%")
                
                # Mostrar probabilidades para todas as classes
                st.markdown("#### 📊 Distribuição de Probabilidades:")
                for i, classe in enumerate(CLASSES):
                    prob = probabilities[i].item()
                    st.progress(prob, text=f"Classe **{classe}**: {prob * 100:.2f}%")
    else:
        st.warning(f"⚠️ Certifica-te de que o ficheiro de pesos `{pesos_path}` está na mesma pasta que o `app.py`.")
