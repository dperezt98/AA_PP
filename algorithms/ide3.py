import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx

class algorithm:
    """
    Permite generar un arbol de decision ide3 mostrando todos los pasos tomados, a partir de un dataset. Ademas de representarlo y predecir nuevos ejemplos
    """

    def __init__(self):
        """
        Constructor de la clase
        """
        # Variables privadas
        self.__decision_node = "decision_node"
        self.__leaf_node = "leaf_node"
        self.__g = None
        self.__root_node = None
        self.__x_attributes = None
        self.__y_attribute = None


    def generate_tree(self, dataset, show):
        """
        Crea un arbol de decision ide3. Dicho arbol sera almacenado en la clase

        Parametros
            dataset: pandas
                Dataset sobre el cual generar el arbol, la ultima columna del mismo sera tomada como la clase de salida
            show: bool
                Cuando su valor es True muestra todos los calculos realizados para generar el arbol
        """
        # Creamos el grafo
        g = nx.Graph()

        # Convertimos todos los elementos del dataset a string
        attribute_list = dataset.columns
        for column in attribute_list:
            dataset[column] = dataset[column].apply(str) # Casteamos todos los valores a string

        # Obtener nombres de los atributos y le nombre de la clase de salida
        x_attribute = np.array(attribute_list[0:-1], dtype='object')
        x_values = [] # Lista de valores unicos para cada atributo
        for x in x_attribute:
            x_values = x_values + [dataset[x].unique()]

        x_values = np.array(x_values, dtype='object')
        self.__x_attributes = x_attribute.copy()

        y_attribute = np.array(attribute_list[-1], dtype='object')
        y_values = dataset[y_attribute].unique() # Lista de valores unicos para definir la salida
        self.__y_attribute = str(y_attribute.copy())

        if show:
            new_show = 1
        else:
            new_show = 0

        root_node = self.__generate_tree_recursive(g, dataset, x_values, y_values, new_show)

        self.__g = g
        self.__root_node = root_node


    def __generate_tree_recursive(self, g, dataset, x_values, y_values, show):
        """
        Genera un arbol de decision IDE3 de forma recursiva

        Parametros
            g: networkx
                Grafo vacio
            dataset: pandas
                Dataset con el cual generar el arbol de decision. La ultima columna de la tabla
                sera seleccionado como el parametro de salida
            x_values: list
                Lista que almacena los tipos de datos encontrados para cada atributo del dataset
            y_values: list
                Lista que almacena el tipo de categoria al que puede pertenecer un ejemplo
            show: int
                Si su valor es 1, muestra las operaciones realizadas por el algoritmo para obtener el arbol
                de decision ide3. Igualar a 0 para no mostrar nada

        Return
            Devuelve una lista de dos valores([arbol_decision, nodo_raiz])
        """
        # Hacer algoritmo recursivo, generaremos un nuevo arbol hasta que los valores de la clase de salida sea iguales
        # para todos los ejemplos


        # Obtener nombres de los atributos y le nombre de la clase de salida
        attribute_list = dataset.columns
        x_attribute = np.array(attribute_list[0:-1], dtype='object')
        y_attribute = np.array(attribute_list[-1], dtype='object')

        # Comprobamos que el atributo y solo tenga un valor valido
        n_values = len(dataset[y_attribute].unique())

        # Si es igual a 1 devolvemos un nodo hoja
        if n_values == 1:
            # Creamos un nuevo nodo de tipo hoja
            value = dataset[y_attribute].unique()[0]
            id_node = self.__add_leaf_node_to(g, value)
            if show > 0:
                self.__show_dataset(show, dataset)
                print("\t" *(show-1), end='')
                print(f"LEAF NODE({value})")
                print("")

            return id_node
        else:
            # Si el numero de atributos es mayor a 1 podemos clasificar
            if (len(x_attribute) > 1):
                if show > 0:
                    print("\t" *(show-1), end='')
                    print(f"========== Decision Table {show} ==========")
                    self.__show_dataset(show, dataset)

                    print("\t" *(show-1), end='')
                    print(f"We must calculate earnings of all attributes from table {show}")

                # Calculamos el atributo de mayor ganancia
                earning_list = []
                for i in range(0, len(x_attribute)):
                    if show > 0:
                        print("\t" *(show-1), end='')
                        print(f"I({x_attribute[i]}) = ", end="")
                        for j in range(0, len(x_values[i])):
                            if j == 0:
                                print(f"I_{x_values[i][j]} ", end="")
                            else:
                                print(f" + I_{x_values[i][j]}", end="")
                        print("")

                    aux_value = self.__attribute_entropy(dataset, x_attribute[i], x_values[i], y_values, show)
                    earning_list += [self.__earning_of(aux_value)]

                    if show > 0:
                        print("\t" *(show-1), end='')
                        print(f"Earning({x_attribute[i]}) = 1 - I({x_attribute[i]}) = 1 - {round(aux_value, 4)} = {round(earning_list[i], 4)}")
                        print("")

                # Obtenemos el mejor atributo
                best_attribute = x_attribute[earning_list == max(earning_list)][0]
                attribute_pos = np.where(earning_list == max(earning_list))[0][0]

                # Obtenemos los valores del mejor atributo
                np_x_values = np.array(x_values, dtype='object')
                best_attribute_values = self.__numpy_to_dict(np_x_values[x_attribute == best_attribute][0])

                # Creamos el nodo del mejor atributo
                id_node = self.__add_decision_node_to(g, best_attribute, best_attribute_values)

                if show > 0:
                    print("\t" *(show-1), end='')
                    for i in range(0, len(x_attribute)):
                        if i == 0:
                            print(f"Earning_list({x_attribute[i]}", end='')
                        else:
                            print(f", {x_attribute[i]}", end='')

                    print(f") = ", end='')
                    for i in range(0, len(x_attribute)):
                        if i == 0:
                            print(f"({round(earning_list[i], 4)}", end='')
                        else:
                            print(f", {round(earning_list[i], 4)}", end='')
                    print(f")")

                    print("\t" *(show-1), end='')
                    print(f"The attribute with higher earning is '{best_attribute}'")
                    print("")

                    print("\t" *(show-1), end='')
                    print(f"DECISION NODE({best_attribute})")
                    show += 1 # Aumentamos la variable para espaciar la informacion

                # Calculamos el arbol resultante a partir de cada valor del atributo. Despues unimos mediante una arista el arbol resultante
                for i in range(0, len(best_attribute_values)):
                    frag_dataset = dataset[dataset[best_attribute] == best_attribute_values[i]]
                    frag_dataset = frag_dataset.drop(columns=[best_attribute]) # Eliminamos la columna de la ganancia maxima

                    if show > 0:
                        print("\t" *(show-1), end='')
                        print(f"When ({best_attribute} = {best_attribute_values[i]}) we get:")

                    # Debemos eliminar los valores del atributo
                    frag_x_values = np.delete(x_values, attribute_pos, axis=0)
                    aux_root_tree_id = self.__generate_tree_recursive(g, frag_dataset, frag_x_values, y_values, show)

                    # Con el id del arbol que cuelga de esta rama, creamos la arista que los conecta
                    g.add_edge(id_node, aux_root_tree_id) # La creamos
                    g.edges[id_node, aux_root_tree_id]['value'] = best_attribute_values[i] # Le asignamos a la rama su valor

                    # Aniadimos al nodo decision la arista creada
                    g.nodes[id_node]['edges'][i] = [id_node, aux_root_tree_id]

                return id_node

            else:
                # Creamos un nuevo nodo de tipo hoja
                value = f"Unknown"
                id_node = self.__add_leaf_node_to(g, value)
                if show > 0:
                    self.__show_dataset(show, dataset)
                    print("\t" *(show-1), end='')
                    print(f"LEAF NODE({value}) - We don't have enough information to classify when '{x_attribute[0]}'={x_values[0]}")
                    print("")

                return id_node

    def draw_tree(self):
        """
        Dibuja el arbol de decision que ha generado el metodo 'generate_tree()' previamente
        """
        if self.__g != None:
            node_labels = {}
            node_colors = []
            for node in self.__g.nodes:
                if len(node_labels) == 0:
                    node_labels[node] = self.__g.nodes[node]["name"] + ' (Root)'
                    node_colors += ['lightcoral']
                else:
                    if self.__g.nodes[node]["type"] == self.__decision_node:
                        node_labels[node] = self.__g.nodes[node]["name"]
                        node_colors += ['lightsteelblue']
                    elif self.__g.nodes[node]["type"] == self.__leaf_node:
                        node_labels[node] = self.__g.nodes[node]["value"]
                        node_colors += ['lightgreen']

            edge_labels = {}
            for edge in self.__g.edges:
                edge_labels[edge] = self.__g.edges[edge]['value']

            pos = nx.spring_layout(self.__g) # Establecemos el tipo de layout de los nodos
            nx.draw(self.__g, pos, labels=node_labels, node_color=node_colors, edge_color='black', arrows = True)
            nx.draw_networkx_edge_labels(self.__g, pos, edge_labels, label_pos=0.5)
            plt.show()

    def predict(self, dataset):
        """
        Predice la clase a la que pertenece los ejemplos facilitados. Se debe haber generado un arbol de decision previamente con el metodo 'generate_tree()'

        Parametros
            dataset: pandas
                Lista de nuevos ejemplos a clasificar. El dataset que contiene los nuevos ejemplos debe tener los mismos atributos que el dataset utilizado para generar el arbol de decision, pero sin contar con la ultima columna que indica la clase. Ejemplo:

                Dataset utilizado para generar el arbol de decision:
                  ----------------------------------------------
                  |Antenas  Colas  Núcleos  Cuerpo|       Clase|
                  ---------------------------------------------|
                  |      1      0        2  Rayado|      Normal|
                  |      1      0        1  Blanco| Cancerígena|
                  |      1      2        0  Rayado|      Normal|
                  |      0      2        1  Rayado|      Normal|
                  |      1      1        1  Rayado| Cancerígena|
                  ----------------------------------------------

                Dataset con nuevos ejemplos a clasificar:
                  ---------------------------------
                  |Antenas  Colas  Núcleos  Cuerpo|
                  ---------------------------------
                  |      1      0        2  Rayado|
                  |      1      1        1  Blanco|
                  |      1      2        0  Blanco|
                  ---------------------------------
        Return
            Devuelve un nuevo dataset con todos los ejemplos clasificados
        """
        check = self.__check_x_attributes_dataset(dataset)
        if self.__g != None and check == True:
            predictions = []
            for i in range(0, len(dataset)):
                predictions += [self.__predict_example(self.__g, dataset.iloc[i])]

            dataset[self.__y_attribute] = predictions
            return dataset
        else:
            raise Exception('Must create a decision tree first')

    def __predict_example(self, g, dataset_row):
        """
        Predice un ejemplo de un dataset

        Parametros
            g: networkx
                Arbol de decision ide3
            dataset_row: Series
                Ejemplo(fila) de un dataset

        Return
            Devuelve la clase a la que pertenece el ejemplo
        """
        node_list = list(g.nodes)
        return self.__predict_example_recursive(g, dataset_row, self.__root_node)

    def __predict_example_recursive(self, g, dataset_row, next_node):
        """
        Recorre el arbol de decision de forma recursiva para determinar la clase a la que pertenece el ejemplo

        Parametros
            g: networkx
                Arbol de decision ide3
            dataset_row: Series
                Ejemplo(fila) de un dataset
            next_node: str
                Id del siguiente nodo a visitar

        Return
            Devuelve la clase a la que pertenece el ejemplo
        """
        node = g.nodes[next_node]
        node_type = node['type']

        if node_type == self.__leaf_node:
            return node['value']
        elif node_type == self.__decision_node:
            attribute_name = node['name']
            branch = dataset_row[attribute_name]

            # Comprobamos si el valor del nuevo ejemplo existe
            if str(branch) in node['values'].values():
                key = self.__get_keys_from_value(node['values'], str(branch))[0]
                edge = node['edges'][key]
                end_edge = edge[1] # Siguiente nodo

                return self.__predict_example_recursive(g, dataset_row, end_edge)
            else:
                raise Exception(f"The value {str(branch)} doesn't exist")

    # ==============================================================================
    # ================================== FORMULAS ==================================
    # ==============================================================================
    def __earning_of(self, entropy_value):
        """
        Calcula la ganancia de un atributo

        Parametros
            entropy_value: float
                Entropia de un atributo

        Return
            Devuelve la ganancia
        """
        value = 1 - entropy_value

        return value

    def __attribute_entropy(self, table, attribute, attribute_values, y_values, show):
        """
        Calcula la entropia que tiene un atributo

        Parametros
            table: pandas
                Dataset sobre el cual calcular la entropia de un atributo
            attribute: str
                Nombre del atributo
            attribute_values: list
                Lista de todos los valores posibles del atributo en el dataset original
            y_values: list
                Lista de todos los valores posibles del atributo y en el dataset original
            show: int
                Variable que indica si se debe mostrar los calculos realizados
        Return
            Devuelve la entropia del atributo
        """
        attribute_column = table[attribute].astype('str').to_numpy()
        total_values = len(table[attribute])
        total_entropy = 0
        value = 0

        if show > 0:
            print("\t" *(show-1), end='')
            print(f"I({attribute}) --------------------------------")

        # I(Ai) = SUM(Nij/N * Iij)
        for i in range(0, len(attribute_values)):
            # Comprobamos la cantidad de veces que aparece el valor attribute_values[i] en la tabla
            n = sum(attribute_column == attribute_values[i])
            value_dataset = table[table[attribute] == attribute_values[i]]

            try:
                if show > 0:
                    print("\t" *(show), end='')
                    print(f"Entropy Attribute value: {attribute_values[i]} - We must calculate his entropy")

                value_entropy = self.__value_attribute_entropy(value_dataset, attribute_values[i], y_values, show)
                value = (n/total_values) * value_entropy

                if show > 0:
                    r_value = round(value, 4)
                    r_value_entropy = round(value_entropy, 4)
                    r_total_entropy = round(total_entropy, 4)

                    print("\t" *(show), end='')
                    print(f"I_{attribute_values[i]} = ", end='')
                    for j in range(0, len(y_values)):
                        if j == 0:
                            print(f"I_{attribute_values[i]}_{y_values[j]} ", end='')
                        else:
                            print(f"+ I_{attribute_values[i]}_{y_values[j]} ", end='')
                    print(f"= {r_value_entropy}")

                    print("\t" *(show-1), end='')
                    print(f"I({attribute}) = {r_total_entropy} + ({n}/{total_values}) * I_{attribute_values[i]} = {r_total_entropy} + {r_value_entropy} = {r_total_entropy + r_value}")

                total_entropy += value

            except:
                if show > 0:
                    r_value = round(value, 4)
                    r_value_entropy = round(value_entropy, 4)
                    r_total_entropy = round(total_entropy, 4)
                    print("\t" *(show-1), end='')
                    print(f"I({attribute}) = {r_total_entropy} + ({n}/{total_values}) * I_{attribute_values[i]} = {r_total_entropy} + {r_value_entropy} = {r_total_entropy + r_value}")

        return total_entropy

    def __value_attribute_entropy(self, attribute_table, attribute, y_values, show):
        """
        Calcula la entropia para un valor determinado de un atributo

        Parametros
            attribute_table: pandas
                Table de ejemplos en la que aparece dicho valor
            attribute: str
                Nombre del atributo
            y_values: list
                Lista de todos los valores posibles del atributo y en el dataset original
            show: int
                Variable que indica si se debe mostrar los calculos realizados

        Return
            Devuelve la entropia del valor
        """
        y_table = attribute_table[attribute_table.columns[-1]] # Tabla con los valores de la columna y
        total_values = len(y_table)

        # Calculamos el numero de veces que aparece cada y_value
        total_entropy = 0

        # Iij = SUM((-1)*(Ni/Nt)*log2(Ni/Nt)) e i=0 hasta Nt
        for i in range(0, len(y_values)):
            n = sum(y_table == y_values[i])

            # Solo realizamos el calculo y n/total_values es distinto de 0. El logaritmo de 0 no existe
            try:
                value = (-1)*(n/total_values)*math.log2(n/total_values)
                total_entropy += value
                if show > 0:
                    print("\t" *(show), end='')
                    print(f"I_{attribute}_{y_values[i]} = (-1)*({n}/{total_values})*log2({n}/{total_values}) = {round(value, 4)}")
            except:
                if show > 0:
                    print("\t" *(show), end='')
                    print(f"I_{attribute}_{y_values[i]} = (-1)*({n}/{total_values})*log2({n}/{total_values}) = 0.0")

        return total_entropy

    # ==============================================================================
    # ============================= METODOS AUXILIARES =============================
    # ==============================================================================
    def __add_decision_node_to(self, g, name, attribute_values):
        """
        Aniade un nodo al grafo g, aniadiendole los valores name y values

        Parametros
            g: networkx
                Grafo al cual aniadir el nodo
            name: str
                Nombre del nodo
            attribute_values: dict
                Lista de valores asociados

        Return
            Devuelve el id del nodo
        """
        id_node = f"{len(g.nodes)}"
        g.add_node(id_node)
        g.nodes[id_node]["type"] = self.__decision_node
        g.nodes[id_node]["name"] = name
        g.nodes[id_node]["values"] = attribute_values # Aniadimos los posibles valores del nodo
        g.nodes[id_node]["edges"] = {}

        return id_node

    def __add_leaf_node_to(self, g, value):
        """
        Aniade un nodo al grafo g, aniadiendole el valor value

        Parametros
            g: networkx
                Grafo al cual aniadir el nodo
            value: str
                Valor del nodo
        Return
            Devuelve el id del nodo
        """
        id_node = f"{len(g.nodes)}"
        g.add_node(id_node)
        g.nodes[id_node]["type"] = self.__leaf_node
        g.nodes[id_node]["value"] = value

        return id_node

    def __numpy_to_dict(self, np_list):
        """
        Convierte un array numpy en un diccionario

        Parametros
            np_list: numpy
                Lista a convertir

        Return
            Devuelve un diccionario
        """
        d = {}
        for i in range(0, len(np_list)):
            d[i] = np_list[i]

        return d

    def __get_keys_from_value(self, d, val):
        """
        Obtiene la clave asociado a un valor

        Parametros
            d: dict
                Diccionario en el que buscar el par clave/valor
            val: object
                Valor a buscar en el diccionario

        Return
            Devuelve una lista con las claves encontradas
        """
        return [k for k, v in d.items() if v == val]

    def __check_x_attributes_dataset(self, dataset):
        """
        Comprueba si los atributos de la variable dataset, coinciden con los atributos del dataset que ha generado el arbol de decision almacenado en la clase

        Parametros
            dataset: pandas
                Dataset a comprobar

        Return
            Devuelve True si los atributos de entrada coinciden
        """
        columns = dataset.columns
        if len(self.__x_attributes) == len(columns):
            for i in range(0, len(self.__x_attributes)):
                if self.__x_attributes[i] != columns[i]:
                    return False
        else:
            return False

        return True

    def __show_dataset(self, show, dataset):
        lines = str(dataset).splitlines()
        width = len(lines[0])

        print("\t" *(show-1), end='')
        print("-"*(width+2))
        for i in range(0, len(lines)):
            if i == 0:
                print("\t" *(show-1), end='')
                print("|", end="")
                print(lines[i], end="")
                print("|")

                print("\t" *(show-1), end='')
                print("-"*(width+2))
            else:
                print("\t" *(show-1), end='')
                print("|", end="")
                print(lines[i], end="")
                print("|")
        print("\t" *(show-1), end='')
        print("-"*(width+2))
        

