#%% 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#%%
class NearestNeighbor:

    def __init__(self):
        self.X = None
        self.Y = None
        self.k = None # k vecinos
    
    def train(self, X, Y):
        self.X = X
        self.Y = Y
        

    def predict(self, X_test, k = 7):
        self.k = k
        assert self.X is not None, 'Train the model first'

        Yp = np.zeros(X_test.shape[0])

        for idx in range(X_test.shape[0]):

            norm = np.linalg.norm(self.X - X_test[idx].ravel(), axis=-1)
            idx_sorted = np.argsort(norm)
            k_nearest = idx_sorted[:self.k]

            Y_preds = self.Y[k_nearest]
            Yp[idx] = np.bincount(Y_preds.astype(int)).argmax()

        return Yp

if __name__ == "__main__":

    train_data = {'class': [], 'vx': [], 'vy': []}
    test_data = {'class': [], 'vx': [], 'vy': []}

    for i in range(100): #distribución parecida a una lineal
        k = np.random.choice(np.arange(1,6))
        vx = k + np.round(np.random.randn()/10, 2) + np.random.choice([0, 0.2])
        vy = k +  np.round(np.random.randn()/10, 2) + np.random.choice([0, 0.2])

        train_data['class'].append(k)
        train_data['vx'].append(vx)
        train_data['vy'].append(vy)

        test_data['class'].append(k)
        test_data['vx'].append(vx - np.random.randn()/2)
        test_data['vy'].append(vy + np.random.randn()/2)

    for key in train_data.keys():
        train_data[key] = np.array(train_data[key])
        test_data[key] = np.array(test_data[key])

    train_ds = pd.DataFrame.from_dict(train_data)
    test_ds = pd.DataFrame.from_dict(test_data)


    train_data2 = {'class': [], 'vx': [], 'vy': []}
    test_data2 = {'class': [], 'vx': [], 'vy': []}

    for i in range(100):  # hice una distribución parecida a la clase "cuadrantes" agregando una clase cerca a los ejes coordenados
        x_sign = np.random.choice([-1, 0, 1])
        y_sign = np.random.choice([-1, 0, 1])
        vx = x_sign + np.random.randn() * 0.1
        vy = y_sign + np.random.randn() * 0.1

        if x_sign == 0 or y_sign == 0:
            k=0
        elif x_sign==1 and y_sign==1:
            k=1
        elif x_sign==-1 and y_sign==1:
            k=2
        elif x_sign==-1 and y_sign==-1:
            k=3
        elif x_sign==1 and y_sign==-1:
            k=4

        train_data2['class'].append(k)
        train_data2['vx'].append(vx)
        train_data2['vy'].append(vy)

        test_data2['class'].append(k)
        test_data2['vx'].append(x_sign + np.random.randn() * 0.1 + np.random.randn()/4)

        test_data2['vy'].append(y_sign + np.random.randn() * 0.1 + np.random.randn()/4)

    for key in train_data2.keys():
        train_data2[key] = np.array(train_data2[key])
        test_data2[key] = np.array(test_data2[key])

    train_ds2 = pd.DataFrame.from_dict(train_data2)
    test_ds2 = pd.DataFrame.from_dict(test_data2)



    #%%
    # Solo tengo que entrenar una vez, puedo mejorar la eficiencia?
    model1 = NearestNeighbor()
    model1.train(train_ds[['vx', 'vy']].values, train_ds['class'].values)

    model2 = NearestNeighbor()
    model2.train(train_ds2[['vx', 'vy']].values, train_ds2['class'].values)
    #%%


    acc1 = []
    acc2 = []

    k_range = np.arange(1, 28, 2)


    for k_neighbor in k_range:

        preds_1 = model1.predict(test_ds[['vx', 'vy']].values, k=k_neighbor)
        preds_2 = model2.predict(test_ds2[['vx', 'vy']].values, k=k_neighbor)

        
        if k_neighbor==7:
            # print(f'Predicciones k=3, dataset A: {preds_1}')
            # print(f'Predicciones k=3, dataset B: {preds_2}')
            preds1_plot = preds_1
            preds2_plot = preds_2

        acc_1 = np.round( 100* np.sum(test_ds['class']==preds_1)/test_ds.shape[0], 2)
        acc_2 = np.round( 100* np.sum(test_ds2['class']==preds_2)/test_ds2.shape[0], 2)

        acc1.append(acc_1)
        acc2.append(acc_2)

    #%%


    # Entrenamiento A
    def plotter(train_ds, test_ds, pred_arr, acc_arr, name):
        
        fig, axes = plt.subplots(1, 3, figsize=(15,5))

        true = axes[0].scatter(train_ds['vx'], train_ds['vy'], c=train_ds['class'], cmap='jet')
        axes[0].set_title(f'Entrenamiento {name}')
        axes[0].set_xlabel('vx')
        axes[0].set_ylabel('vy')
        legend1 = axes[0].legend(*true.legend_elements(), title="Classes", loc="center right")
        axes[0].add_artist(legend1)

        # Test A
        test = axes[1].scatter(test_ds['vx'], test_ds['vy'], c=test_ds['class'], cmap='jet', marker='s', label='True')
        preds2 = axes[1].scatter(test_ds['vx'], test_ds['vy'], c=pred_arr, cmap='jet', marker='x', label='Predicted')
        axes[1].set_title(f'Test {name}, k=7')
        axes[1].set_xlabel('vx')
        axes[1].set_ylabel('vy')
        legend_true = axes[1].legend(*test.legend_elements(), title="True", loc="upper left")
        axes[1].add_artist(legend_true)
        legend_pred = axes[1].legend(*preds2.legend_elements(), title="Predicted", loc="lower right")
        axes[1].add_artist(legend_pred)

        ev = axes[2].plot(k_range, acc_arr, marker='o')
        axes[2].set_title(f'Accuracy {name}')
        axes[2].set_xlabel('k vecinos')
        axes[2].set_ylabel('Accuracy (%)')
        # axes[2].set_xticks(k_range)
        axes[2].set_yticks(np.arange(50, 105, 10))
        axes[2].grid(True)

        plt.tight_layout()
        # plt.savefig(f'ej1_zuniga_{name}.png', dpi=300)
        plt.show()

    # %%

    plotter(train_ds, test_ds, preds1_plot, acc1, 'A')
    plotter(train_ds2, test_ds2, preds2_plot, acc2, 'B')
    # %%
    # Grafiquemos frontera de decisiónl rango [0,5]
    random_points1 = np.random.rand(1000, 2) * 5  # puntos en el rango [0, 5]x[0, 5]
    random_preds1 = model1.predict(random_points1, k=7)
    random_points2 = np.random.rand(1000, 2)*3 - 1.5 # puntos en el rango [-1.5, 1.5]
    random_preds2 = model2.predict(random_points2, k=7)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(random_points1[:,0], random_points1[:,1], c=random_preds1, cmap='jet', alpha=0.5)
    plt.title('Frontera de decisión A, k=7')

    plt.subplot(1,2,2)
    plt.scatter(random_points2[:,0], random_points2[:,1], c=random_preds2, cmap='jet', alpha=0.5)
    plt.title('Frontera de decisión B, k=7')
    # plt.savefig('ej1_zuniga_frontera.png', dpi=300)
    plt.show()
    # %%
