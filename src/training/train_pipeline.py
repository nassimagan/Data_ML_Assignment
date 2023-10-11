import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from src.constants import RAW_DATASET_PATH, MODELS_PATH, REPORTS_PATH, LABELS_MAP
from src.models.naive_bayes_model import NaiveBayesModel
from src.models.svc_model import SVCModel
from src.models.xgbc_model import XGBCModel
from src.utils.plot_utils import PlotUtils
from sklearn.feature_extraction.text import TfidfVectorizer


class TrainingPipeline:
    def __init__(self):
        df = pd.read_csv(RAW_DATASET_PATH)

        text = df['resume']
        y = df['label']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            text,
            y,
            test_size=0.2,
            random_state=0
        )

        self.model = None

    def train(self, serialize: bool = True, model_name: str = 'model'):

        if model_name == 'Naive Bayes':
            self.model = NaiveBayesModel()
        elif model_name == 'SVC':
            self.model = SVCModel()
        else:
            self.model = XGBCModel()


        
        # Fit the model 
        self.model.fit(self.x_test, self.y_train)

        model_path = MODELS_PATH / f'{model_name}.joblib'
        if serialize:
            self.model.save(model_path)

    def get_model_perfomance(self) -> tuple:
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions), f1_score(self.y_test, predictions, average='weighted')

    def render_confusion_matrix(self, plot_name: str = 'cm_plot'):
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        plt.rcParams['figure.figsize'] = (14, 10)

        PlotUtils.plot_confusion_matrix(
            cm,
            classes=list(LABELS_MAP.values()),
            title='Naive Bayes'
        )

        plot_path = REPORTS_PATH / f'{plot_name}.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    tp = TrainingPipeline()
    tp.train(serialize=True)
    accuracy, f1_score = tp.get_model_perfomance()
    tp.render_confusion_matrix()
    print(f'ACCURACY = {accuracy}, F1 SCORE = {f1_score}')
