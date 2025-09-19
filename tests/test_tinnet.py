from tinnet import Tinnet
import torch

def test_tinnet_predict():
    # Create a dummy model and save it to mimic real use
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=1, keepdim=True)
    dummy_model = {
        "adsorption_energy": DummyModel(),
        "cohesive_energy": DummyModel(),
        "band_center": DummyModel(),
        "band_moments": DummyModel()
    }
    torch.save(dummy_model, "tinnet/models/tinnet/pretrained_weights.pt")

    model = Tinnet(property="adsorption_energy")
    x = torch.ones((2, 4))
    y = model.predict(x)
    assert y.shape == (2, 1)
    print("Test passed: Tinnet predict")
