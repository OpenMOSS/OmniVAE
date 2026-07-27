from audiobox_aesthetics.infer import initialize_predictor

class AudioBoxInferencer:
    def __init__(self, model_path):
        self.predictor = initialize_predictor(ckpt=f"{model_path}/audiobox-aesthetics/checkpoint.pt")

    def infer(self,audio_path):
        score = self.predictor.forward([{"path":audio_path}])
        return score[0]



