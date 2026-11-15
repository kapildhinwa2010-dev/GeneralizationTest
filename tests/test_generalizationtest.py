import pytest
import torch
from generalizationtest import GeneralizationTest, create_generalizationtest

class TestGeneralizationTest:
    def test_init(self):
        m = GeneralizationTest(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = GeneralizationTest(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_generalizationtest()
        assert isinstance(m, GeneralizationTest)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
