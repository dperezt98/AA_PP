import numpy as np
import matplotlib.pyplot as plt


class algorithm:

    def __init__(self):
        # Variables privadas
        pass

    def prepare_dataset(self, dataset):
        """
        Prepara el dataset para su utilización. Aniade la columna X0 al cojunto de datos
        
        Parametros
            dataset: pandas
                Dataset a preparar. Debe tener 2 columnas, una para lo valores de 'x' y otra para los de 'y'
                
        Return
            Devuelve una lista de dos listas([X, Y])). En la primera se encuentra el conjunto de valores de entrada actualizados y en la segunda la lista de valores de salida
        """
        if len(dataset.columns) == 2:
            columns = dataset.columns
            columns_names = columns[:]

            X = dataset[columns_names[0]].to_numpy()
            X = X.reshape((len(X), 1))
            Y = dataset[columns_names[1]].to_numpy()

            # Aniadimos X0 = 1
            X = self.__add_X0(X)

            return X, Y
        else:
            raise Exception("The dataset has more than 2 columns")

    def calculate_theta(self, X, Y, show):
        """
        Nos permite calcular los valores de theta para construir la recta de regresion
        
        Parametros
            X: numpy
                Matriz de valores de entrada. Deben haber sido preparados con la funcion 'prepare_dataset()' previamente
            Y: numpy
                Lista de valores de entrada
            show: boolean
                Si es True nos mostrara todos los calculos realizados para obtener los valores de theta
                
        Return
            Devuelve una lista con los dos valores de theta
        """
        if show:
            print("-------------- Formulas --------------")
            print("Θ = inv(X_t * X) * X_t * Y")
            print("--------------------------------------")
            print("")

        # A = X_t * X
        X_t = np.transpose(X)
        A = np.round(np.matmul(X_t, X), 4)
        if show:
            print(f"A = X_t * X = [{A[0][0]} {A[0][1]}]")
            print(f"              [{A[1][0]} {A[1][1]}]")
            print("")

        # inv(A) // det(A)
        A_det = np.round(np.round(np.linalg.det(A), 4))
        A_t = np.transpose(A)
        A_cof = self.__cofactor_matrix(A_t)
        A_inv = None
        if A_det != 0:
            A_inv = np.round((A_cof / A_det), 4)

        if show:
            print(f"-----------inv(A)-----------")
            print(f"inv(A) = (1/det(A)) * Cofactor(A_t)")
            print(f"")
            print(f"A = [a b] | det(A) = a*d - b*c = {A[0][0]}*{A[1][1]} - {A[0][1]}*{A[1][0]} = {A_det}")
            print(f"    [c d]")
            print(f"")
            if A_det == 0:
                print("If det(A) is equal to 0, inv(A) doesn't exist")
                raise Exception("inv(A) doesn't exist")
            print(f"A_t = [{A_t[0][0]} {A_t[0][1]}]")
            print(f"      [{A_t[1][0]} {A_t[1][1]}]")
            print(f"")
            print(f"Cof(A_t) = [d -c] = [{A[1][1]} {-1*A[1][0]}]")
            print(f"           [-b a]   [{-1*A[0][1]} {A[0][0]}]")
            print(f"")
            print(f"inv(A) = 1/{A_det} * [{A[1][1]} {-1*A[1][0]}] = [{A_inv[0][0]} {A_inv[0][1]}]")
            print(f"                  [{-1*A[0][1]} {A[0][0]}]    [{A_inv[1][0]} {A_inv[1][1]}]")
            print(f"----------------------------")

        # B = A_inv * X_t
        B = np.round(np.matmul(A_inv, X_t), 4)
        if show:
            print(f"Θ = inv(A) * X_t * Y = B * Y")
            print("")
            print(f"B = inv(A) * X_t = ")
            print(f"{B}")
            print("")

        # theta = B * Y
        theta = np.matmul(B, Y)
        theta = np.round(theta, 4)
        if show:
            print(f"Θ = B * Y")
            print(f"")
            print(f"Θ = {theta}")

        self.__theta = theta.copy()
        return np.round(theta, 4)

    def lost_function(self, X, Y, theta, show):
        """
        Calcula el valor de la funcion de costo
        
        Parametros
            X: numpy
                Matriz de valores de entrada. Deben haber sido preparados con la funcion 'prepare_dataset()' previamente
            Y: numpy
                Lista de valores de entrada
            theta: list
                Lista de los dos valores de theta
            show: boolean
                Si es True nos mostrara todos los calculos realizados para obtener los valores de theta
                
        Return
            Valor de la funcion de costo
        """

        if show:
            print("-------------- Formulas --------------")
            print(f"J(Θ) = (1/2) * SUM_j_to_m[ (h_theta(x_j) - y_j)^2 ]")
            print(f"hθ(x) = SUM_j_to_m[ x_j *  θ_j]")
            print("--------------------------------------")
            print("")

        amount_list = []
        for i in range(0, len(X)):
            amount_list += [(self.__h0(X[i], theta) - Y[i])**2]
            
            if show:
                if i == 0:
                    print(f"J(Θ) = (1/2) * [([{X[i][0]}*{theta[0]} + {X[i][1]}{theta[1]}] - {Y[i]})^2 ", end='')
                else:
                    print(f"+ ([{X[i][0]}*{theta[0]} + {X[i][1]}{theta[1]}] - {Y[i]})^2 ", end='')

        print(f"]")

        amount = sum(amount_list)/2
        
        if show:
            print(f"J(Θ) = (1/2) * [", end='')
            for i in range(0, len(amount_list)):
                if i == 0:
                    print(f"{amount_list[i]}", end='')
                else:
                    print(f" + {amount_list[i]}", end='')

            print(f"] = {amount}")
            print("")
            print(f"J(Θ) = {amount}")

        return amount

    def gradient_descent(self, X, Y, theta, alpha, iterations, show):
        """
        Calcula la estimacion de los valores de theta mediante el algoritmo de descenso por gradiente
        
        Parametros
            X: numpy
                Matriz de valores de entrada. Deben haber sido preparados con la funcion 'prepare_dataset()' previamente
            Y: numpy
                Lista de valores de entrada
            theta: list
                Lista de los dos valores de theta iniciales
            iterations: int
                Numero de veces que se aplicara el descenso por gradiente sobre los valores de theta
            show: boolean
                Si es True nos mostrara todos los calculos realizados para obtener los valores de theta
        """
        if show:
            print("-------------- Formulas --------------")
            print("θj' = θj + α * Σ_i_to_m[ (h_θ(x_i) - y_i) * x_ij ]")
            print("--------------------------------------")
            print("")

        theta_prima = np.zeros(len(theta))
        theta_aux = theta.copy()
        
        # Iteracion
        for n in range(0, iterations):
            if show:
                print(f"-------------- Iteration {n+1} --------------")

            # Calculamos theta_i
            for i in range(0, len(theta_aux)):
                if show:
                    print(f"θ{i}' = θ{i} + ({alpha})[", end='')
                
                aux_list = []
                # Calculamos el sumatorio
                for j in range(0, len(X)):
                    aux_list += [round( (self.__h0(X[j], theta_aux) - Y[j]) * X[j][i], 4)]
                    theta_prima[i] = round(theta_aux[i] + alpha * sum(aux_list), 4)
                    
                    if show:
                        if j == 0:
                            print(f" ([{X[j][0]}*{theta_aux[0]} + {X[j][1]}*{theta_aux[1]}] - {Y[i]}) * {X[j][i]} ", end='')
                        else:
                            print(f" + ([{X[j][0]}*{theta_aux[0]} + {X[j][1]}*{theta_aux[1]}] - {Y[i]}) * {X[j][i]} ", end='')
              
                if show:
                    print("]")
                    print(f"θ{i}' = θ{i} + ({alpha})[", end='')
                    for a in range(0, len(aux_list)):
                        if a == 0:
                            print(f" {aux_list[a]}", end='')
                        else:
                             print(f" + {aux_list[a]} ", end='')
                                   
                    print(f"] = {theta_prima[i]}")
                    print("") 
                
            theta_aux = np.round(theta_prima.copy(), 4)

            if show:
                print(f"[θ0, θ1] = [{theta_prima[0]}, {theta_prima[1]}]")
                print("")
        return theta_prima

    def represent_dataset(self, dataset):
        """
        Representa los datos de un dataset
        
        Parametros
            dataset: pandas
                Dataset a representar. Debe tener 2 columnas, una para lo valores de 'x' y otra para los de 'y'
        """
        X, Y = self.prepare_dataset(dataset)
        X = X[:, 1]
        columns = dataset.columns
        columns_names = columns[:]

        # Representamos
        ax = plt.figure()
        ax.set_facecolor('white')

        plt.scatter(X, Y, color='r', zorder=3)

        # Leyenda
        plt.xlabel(f"{columns_names[0]}")
        plt.ylabel(f"{columns_names[1]}")
        plt.grid(zorder=0)
        plt.show()

    def represent_regresion(self, dataset, theta):
        """
        Representa los datos de un dataset y una recta de regresion
        
        Parametros
            dataset: pandas
                Dataset a representar. Debe tener 2 columnas, una para lo valores de 'x' y otra para los de 'y'
            theta: list
                Lista de los dos valores de theta utilizados para generar la recta de regresion
        """
        X, Y = self.prepare_dataset(dataset)
        columns = dataset.columns
        columns_names = columns[:]
        self.__represent(X, Y, theta, columns_names)

    def __represent(self, X, Y, theta, columns_names):
        """
        Representa un conjunto de datos y su recta de regresion
        
        Parametros
            X: numpy
                Matriz de valores de entrada. Deben haber sido preparados con la funcion 'prepare_dataset()' previamente
            Y: numpy
                Lista de valores de entrada
            theta: list
                Lista de los dos valores de theta iniciales
            columns_names: list
                Lista de dos valores. Contiene el nombre del tipo de datos de 'x' y de 'y'
        """
        # Preparamos los datos
        aux_X = X[:, 1].copy()

        x_max = max(aux_X)
        x_min = min(aux_X)
        y_max = max(Y)
        y_min = min(Y)

        if (x_max - y_min) <= 1:
            line_x = np.arange(x_min, x_max+0.05, 0.05)
        else:
            line_x = np.arange(x_min, x_max+1)

        line_y = []
        for x in line_x:
            line_y += [self.y(theta, x, False)]

        # Representamos
        ax = plt.figure()
        ax.set_facecolor('white')

        plt.scatter(aux_X, Y, color='r', zorder=3)
        plt.plot(line_x, line_y, zorder=2)
        legend = [f"y(x) = {theta[0]} + {theta[1]} * x"]

        # Leyenda
        plt.xlabel(f"{columns_names[0]}")
        plt.ylabel(f"{columns_names[1]}")
        plt.grid(zorder=0)
        # plt.ylim((0,1000))
        plt.legend(legend)
        plt.show()

    # ==============================================================================
    # ================================== FORMULAS ==================================
    # ==============================================================================
    def __h0(self, X_i, theta):
        """
        Calcula la hipotesis resultante para los valores de X_i y los de theta
        
        Parametros
            X_i: list
                Lista de los valores de X del ejemplo i
            theta: list
                Valores de theta
                
        Return
            Resultado de la hipotesis
        """
        return round(np.matmul(X_i, theta), 4)

    def __cofactor_matrix(self, A):
        """
        Calcula la matriz adjunta de A
        
        Parametros
            A: numpy
                Matriz a la cual calcular su adjunta
                
        Return 
            Devuelve la matriz adjunta
        """
        # Adj(A) = transpose(inv(A)) * det(A)
        cofactor = None  # Adjunta
        cofactor = np.linalg.inv(A).T * np.linalg.det(A)

        # return cofactor matrix of the given matrix
        return cofactor

    def y(self, theta, x, show):
        """
        Devuelve el valor de salida de una recta de regresion. y(x) = theta[0] + theta[1] * x
        
        Parametros
            theta: list
                Lista de dos valores de theta empleados en la funcion de regresion
            x: float
                Valor de entrada
                
        Return
            Valor de obtenido por la funcion
        """
        y = theta[0] + theta[1]*x
        if show:
            print(f"y({x}) = {theta[0]} + {theta[1]} * {x} = {round(y, 4)}")
        return round(y, 4)

    # ==============================================================================
    # ============================= METODOS AUXILIARES =============================
    # ==============================================================================
    def __add_X0(self, X):
        """
        Aniade una columna de '1' al conjunto de datos de X
        
        Parametros
            X: numpy
                Lista de valores
        
        Return
            Devuelve el mismo conjunto de datos pero con una columna de '1' al principio de la lista
        """
        # Aniadimos una fila de unos
        X0 = np.ones(len(X)).reshape((len(X), 1))
        new_X = np.append(X0, X, axis=1)
        return new_X
