AstroSurface AI
Conditional GAN-Based Hypothetical Lunar Impact Scenario Generation
AstroSurface AI is a Python prototype that generates illustrative
synthetic crater-like terrain images using a Conditional Deep
Convolutional Generative Adversarial Network (Conditional DCGAN). A
Streamlit web interface lets a user select a scenario label---Small,
Medium, or Large---and generate an image that can be downloaded as a
PNG.
Important limitation: The training images are procedurally
generated synthetic terrain, not real lunar observations. The outputs
are illustrative only; they do not simulate impact physics or predict
real lunar impacts. The current model may produce blurry images, and
scenario differences may be limited.

Features
- Creates a labelled synthetic dataset for Small, Medium, and Large
  crater scenarios.
- Trains a conditional GAN using PyTorch.
- Saves generator and discriminator model weights.
- Provides a Streamlit web app for image generation.
- Supports PNG image downloads.