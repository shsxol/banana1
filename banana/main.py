import os
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
import torch.nn as nn
from keras.models import load_model

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the ResDown class
class ResDown(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.batch_norm1 = nn.BatchNorm2d(channel_out//2, 0.8)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, scale, 1)
        self.batch_norm2 = nn.BatchNorm2d(channel_out, 0.8)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, scale, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        skip = self.conv3(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x + skip)

        return x

# Define the ResUp class
class ResUp(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.batch_norm1 = nn.BatchNorm2d(channel_out//2, 0.8)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.batch_norm2 = nn.BatchNorm2d(channel_out, 0.8)

        self.upscale = nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        skip = self.conv3(self.upscale(x))

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)

        x = self.conv2(self.upscale(x))
        x = self.batch_norm2(x)

        x = self.activation(x + skip)

        return x
# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, channels, ch=64, z=512):
        super(Encoder, self).__init__()
        self.conv1 = ResDown(channels, ch)
        self.conv2 = ResDown(ch, 2*ch)
        self.conv3 = ResDown(2*ch, 4*ch)
        self.conv4 = ResDown(4*ch, 8*ch)
        self.conv5 = ResDown(8*ch, 8*ch)
        self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)
        self.conv_log_var = nn.Conv2d(8*ch, z, 2, 2)

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        x = self.sample(mu, log_var)

        return x, mu, log_var

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, channels, ch=64, z=512):
        super(Decoder, self).__init__()
        self.conv1 = ResUp(z, ch*8)
        self.conv2 = ResUp(ch*8, ch*4)
        self.conv3 = ResUp(ch*4, ch*2)
        self.conv4 = ResUp(ch*2, ch)
        self.conv5 = ResUp(ch, ch//2)
        self.conv6 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return self.activation(x)
# Define the ResnetGenerator class
class ResnetGenerator(nn.Module):
    def __init__(self, channel_in=3, ch=64, z=512):
        super(ResnetGenerator, self).__init__()
        self.encoder = Encoder(channel_in, ch=ch, z=z)
        self.decoder = Decoder(channel_in, ch=ch, z=z)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon = self.decoder(encoding)
        return recon, mu, log_var
# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize, dropout, spectral):
            """Returns layers of each discriminator block"""
            if spectral:
                layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, stride, 1), n_power_iterations=2)]
            else:
                layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(p=0.5))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize, dropout, spectral in [(64, 2, False, 0, 0), (128, 2, True, 0, 0), (256, 2, True, 0, 0), (512, 1, True, 0, 0)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize, dropout, spectral))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
# Load the saved checkpoint
checkpoint = torch.load('painting_model.pth', map_location=torch.device('cpu'))


# Extract the generator and discriminator models
generator = checkpoint['generator']
discriminator = checkpoint['discriminator']

# Set both models to evaluation mode
generator.eval()
discriminator.eval()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def processImage(filename, operation):
    print(f"the operation is {operation} and the filename is {filename}")
    img = cv2.imread(f"uploads/{filename}")
    
    if operation == "1":
        # Perform image inpainting using the generator model
        with torch.no_grad():
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            img_processed, _, _ = generator(img_tensor)
            img_processed = (img_processed.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite(f"static/{filename}", img_processed)
        return filename
    
    elif operation == "2":
        # Define noise or pass it as an argument
        noise = np.random.normal(loc=0, scale=1, size=img.shape)
        # Perform image processing using the discriminator model
        with torch.no_grad():
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            img_processed = discriminator(img_tensor)
            img_processed = (img_processed.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite(f"static/{filename}", img_processed)
        return filename
    
    else:
        return None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/edit', methods=["GET", "POST"])
def edit():
    if request.method == "POST":
        operation = request.form.get("operation")
        if 'file' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return "error no selected file "
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Image processing function
            processImage(filename, operation)
            flash(f"Your image has been processed and is available <a href='/static/{filename}' target='_blank'> here</a>")
            return render_template("index.html")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
