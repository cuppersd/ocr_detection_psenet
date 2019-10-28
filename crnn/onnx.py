import onnxruntime
import cv2
import numpy as np

from .charset import alphabetChinese


class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def decode(self, np_t, np_length, soft_max):
        if len(np_length) == 1:
            length = np_length[0]
            t = np_t[:length]
            char_list = []
            score = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
                    if soft_max:
                        score.append(soft_max[i][t[i]])
            if score:
                return score, ''.join(char_list)
            else:
                return ['None'], ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(np_length.numel()):
                l = np_length[i]
                texts.append(self.decode(np_t[index:index + l], [l], soft_max))
                index += l
            return texts


class Crnn(object):
    def __init__(self, model_path, cal_score = False):
        self.decoder = strLabelConverter(alphabetChinese)
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.cal_score = cal_score

    def norm_image(self, image):
        h, w = image.shape
        scale = h * 1.0 / 32
        w = int(w / scale)
        image = cv2.resize(image, (w, 32), interpolation=cv2.INTER_CUBIC) / 255
        image = (image - 0.5) / 0.5
        image = image.reshape(1, 1, 32, -1).astype(np.float32)  # [batch, 32, w]
        return image


    def recognize(self, image):
        '''
        :param image: numpy gray
        :return: confidence and text
        '''
        ort_inputs = {self.ort_session.get_inputs()[0].name: self.norm_image(image)}
        ort_outputs = self.ort_session.run(None, ort_inputs)[0]
        scores = []
        sim_preds = []
        for i in range(ort_outputs.shape[1]):
            preds = ort_outputs[:, i, :].reshape(-1, 1, 5533)
            soft_max = list()
            if self.cal_score:
                for pred in preds:
                    assert pred.shape == (1, 5533), u'shape is not (1, 5533):{}'.format(
                        pred.shape)
                    pred = np.array(pred[0], np.float64)
                    soft_max.append(np.exp(pred) / sum(np.exp(pred)))
            np_pred_ = np.argmax(preds, axis=2)
            np_pred_ = np.squeeze(np_pred_)
            np_pred_size = np_pred_.shape
            score, sim_pred = self.decoder.decode(np_pred_, np_pred_size, soft_max)
            scores.append(score)
            sim_preds.append(sim_pred)

        return scores, sim_preds




if __name__ == '__main__':
    import time
    image = cv2.imread('22.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    crnn = Crnn('crnn.onnx')
    t = time.time()
    scores, sim_preds = crnn.recognize(image)
    print(time.time() - t)
    print(scores, sim_preds)
