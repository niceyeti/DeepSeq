from unittest import TestCase
import architecture


class TestArchitecture(TestCase):
    def test_foo(self):
        self.assertTrue(True)

    def test_inference(self):
        ys = architecture.inference_test()
        self.assertTrue(ys is not None, "Inference check passes if no errors raised")

    def test_encoder_inference(self):
        ys = architecture.encoder_inference_test()
        self.assertTrue(ys is not None, "Inference check passes if no errors raised")
