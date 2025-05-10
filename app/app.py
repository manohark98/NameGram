import os
import logging
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# ----------------------
# Define custom model layers
# ----------------------
class LinearLayer:
    def __init__(self, f_in, f_out, bias=True) -> None:
        self.f_in = f_in
        self.f_out = f_out
        self.bias = bias
        self.w = torch.randn((self.f_in, self.f_out), requires_grad=True)
        if self.bias:
            self.b = torch.zeros(self.f_out, requires_grad=True)
        else:
            self.b = None

    def __call__(self, x):
        return x @ self.w + self.b if self.bias else x @ self.w

    def parameters(self):
        return [self.w] if self.b is None else [self.w, self.b]

    def state_dict(self):
        return {'weight': self.w, 'bias': self.b} if self.bias else {'weight': self.w}

    def load_state_dict(self, state):
        self.w.data.copy_(state['weight'])
        if self.bias:
            self.b.data.copy_(state['bias'])

class EmbeddingLayer:
    def __init__(self, n_emb, d_emb) -> None:
        self.emb = torch.randn((n_emb, d_emb), requires_grad=True)

    def __call__(self, x):
        return self.emb[x.long()]

    def parameters(self):
        return [self.emb]

    def state_dict(self):
        return {'embeddings': self.emb}

    def load_state_dict(self, state):
        self.emb.data.copy_(state['embeddings'])

class FlattenConsecutive:
    def __init__(self, n) -> None:
        self.n = n
    def __call__(self, x):
        i, j, k = x.shape
        x = x.view(i, j//self.n, k*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x
    def parameters(self):
        return []

class Tanh:
    def __call__(self, x):
        return torch.tanh(x)
    def parameters(self):
        return []

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean.view(1, -1) if x.ndim == 2 else self.running_mean.view(1,1,-1)
            xvar = self.running_var.view(1, -1) if x.ndim == 2 else self.running_var.view(1,1,-1)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        return out
    def parameters(self):
        return [self.gamma, self.beta]
    def state_dict(self):
        return {'gamma': self.gamma, 'beta': self.beta, 'running_mean': self.running_mean, 'running_var': self.running_var}
    def load_state_dict(self, state):
        self.gamma.data.copy_(state['gamma'])
        self.beta.data.copy_(state['beta'])
        self.running_mean.data.copy_(state['running_mean'].squeeze())
        self.running_var.data.copy_(state['running_var'].squeeze())

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def state_dict(self):
        state = {}
        for idx, layer in enumerate(self.layers):
            state[f"layer_{idx}"] = layer.state_dict() if hasattr(layer, 'state_dict') else {}
        return state
    def load_state_dict(self, state):
        for idx, layer in enumerate(self.layers):
            key = f"layer_{idx}"
            if hasattr(layer, 'load_state_dict') and key in state:
                layer.load_state_dict(state[key])
    def to(self, device):
        for layer in self.layers:
            for p in layer.parameters():
                p.data = p.data.to(device)
            if isinstance(layer, BatchNorm1d):
                layer.running_mean = layer.running_mean.to(device)
                layer.running_var = layer.running_var.to(device)
        return self

# ----------------------
# Build and load the trained model
# ----------------------
device = torch.device('cpu')
vocab_size = 27
block_size = 16
itos = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'g',8:'h',9:'i',10:'j',11:'k',12:'l',13:'m',14:'n',15:'o',16:'p',17:'q',18:'r',19:'s',20:'t',21:'u',22:'v',23:'w',24:'x',25:'y',26:'z',0:'.'}
n_embd = 80
n_hidden = 200
model = Sequential([
    EmbeddingLayer(vocab_size, n_embd),
    FlattenConsecutive(2), LinearLayer(n_embd*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), LinearLayer(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), LinearLayer(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), LinearLayer(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    LinearLayer(n_hidden, vocab_size)
]).to(device)
# Load weights
state = torch.load('hindu_names.pth', map_location=device)
model.load_state_dict(state)
# Set eval
for layer in model.layers:
    if hasattr(layer, 'training'):
        layer.training = False
model = model  # ready for inference

# ----------------------
# Setup Flask app
# ----------------------
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
CORS(app)

# About Engine Information
NAME_ENGINE = {
    "model_type": "MLP",
    "parameters_count": 281187,
    "description": "Multi-Layer Perceptron neural network for name generation"
}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/generate-name', methods=['GET'])
def generate_name():
    # generate a name using the trained model
    out = []
    context = [0] * block_size
    while True:
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(ix)
    name = ''.join(itos[i] for i in out)
    #logging.debug(f"Generated name: {name}")
    return jsonify({'name': name})

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG)
    app.run(host='127.0.0.1', port=8000, debug=True)
