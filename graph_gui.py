from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QTableWidget, QTableWidgetItem, QSpinBox, QLabel, QPushButton
#from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QSize, Qt
import numpy as np
import networkx as nx
#import matplotlib
#matplotlib.use('Qt5Agg')
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def map_click_event(event, x, y, flags, params):
    global nodes_count_user
	# checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        nodes_count_user += 1

        pos[len(pos)] = np.array([x, y], dtype=np.int64)
        
        cv2.circle(img,(x,y),8,(255,0,0),-1)
        cv2.putText(img, str(nodes_count_user - 1), (x,y), cv2.FONT_HERSHEY_SIMPLEX,
		            0.5, (0, 0, 0), 1)
        cv2.imshow('image', img)
 
class MainWindow(QMainWindow):
    """Наследуемся от QMainWindow"""

    nodes_count = 0     # количество вершин (= размер матрицы смежности adjacency matrix)
    nodes_coords = dict()   # координаты вершин
    G = None   # граф nx.DiGraph
    adj_matrix = None   # матрица инцидентности (numpy nd array dtype=int64)

    def __init__(self, nodes_count_user, pos):
        """Переопределяем конструктор класса"""
        # Обязательно нужно вызвать метод супер класса
        QMainWindow.__init__(self)
        self.nodes_count = nodes_count_user
        self.nodes_coords = pos
        for node in self.nodes_coords:
            self.nodes_coords[node][1] = abs(703 - self.nodes_coords[node][1])
        self.setMinimumSize(QSize(480, 400))             # Устанавливаем размеры
        self.setWindowTitle("Построитель графов")    # Устанавливаем заголовок окна
        central_widget = QWidget(self)                  # Создаём центральный виджет
        self.setCentralWidget(central_widget)           # Устанавливаем центральный виджет
 
        grid_layout = QGridLayout()             # Создаём QGridLayout
        central_widget.setLayout(grid_layout)   # Устанавливаем данное размещение в центральный виджет

        self.label_nodes_count = QLabel('Задайте количество вершин матрицы:', self)
        grid_layout.addWidget(self.label_nodes_count, 0, 0)
        
        self.spinbox = QSpinBox()
        grid_layout.addWidget(self.spinbox, 0, 1)   # Добавляем спинбокс в сетку

        button_matr_size = QPushButton('Применить', self)
        grid_layout.addWidget(button_matr_size, 0, 2)
        button_matr_size.clicked.connect(self.button_matr_size_clicked)

        # надпись "матрица смежности"
        self.label_adj_matrix = QLabel('Матрица смежности:', self)
        grid_layout.addWidget(self.label_adj_matrix, 1, 0)
        self.label_adj_matrix.setMaximumHeight(30)
        self.qt_adj_matrix = QTableWidget(self.nodes_count, self.nodes_count, self)  # Создаём матрицу смежности
        #self.qt_adj_matrix.setMinimumSize(QSize(100, 100)) 
        self.qt_adj_matrix.setMinimumHeight(100)
        
        # Устанавливаем заголовки таблицы
        """ table.setHorizontalHeaderLabels(["Header 1", "Header 2", "Header 3"])
 
        # Устанавливаем всплывающие подсказки на заголовки
        table.horizontalHeaderItem(0).setToolTip("Column 1 ")
        table.horizontalHeaderItem(1).setToolTip("Column 2 ")
        table.horizontalHeaderItem(2).setToolTip("Column 3 ")
 
        # Устанавливаем выравнивание на заголовки
        table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignLeft)
        table.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)
        table.horizontalHeaderItem(2).setTextAlignment(Qt.AlignRight) """
        
        # заполняем матрицу нулями
        for i in range(self.qt_adj_matrix.rowCount()):
            for j in range(self.qt_adj_matrix.columnCount()):
                self.qt_adj_matrix.setItem(i, j, QTableWidgetItem(str(0)))
        self.qt_adj_matrix.resizeColumnsToContents()	# делаем ресайз колонок по содержимому
        grid_layout.addWidget(self.qt_adj_matrix, 2, 0, 2, 1)   # Добавляем матрицу в сетку

        self.label_adj_tutorial = QLabel('Нажмите на кнопку "Рандом", чтобы заполнить\n матрицу смежности случайным образом\n либо заполните её самостоятельно, после\n чего создайте структуру графа по кнопке:', self)
        grid_layout.addWidget(self.label_adj_tutorial, 2, 1, 1, 2)

        button_random_adj = QPushButton('Рандом', self) # кнопка заполнения матрицы adj рандомно
        grid_layout.addWidget(button_random_adj, 3, 1)
        button_random_adj.clicked.connect(self.button_random_adj_clicked)

        button_create_fr_adj = QPushButton('Создать граф', self) # создаём структуру графа и заполняем остальные таблицы автоматически
        grid_layout.addWidget(button_create_fr_adj, 3, 2)
        button_create_fr_adj.clicked.connect(self.button_create_fr_adj_clicked)

        # надпись "матрица инцидентности"
        self.label_inc_matrix = QLabel('Матрица инцидентности:', self)
        self.label_inc_matrix.setMaximumHeight(30)
        grid_layout.addWidget(self.label_inc_matrix, 4, 0)

        # матрица инцидентности
        self.qt_inc_matrix = QTableWidget(self.nodes_count, 4, self)  # Создаём таблицу
        self.qt_inc_matrix.setMinimumSize(QSize(200, 200)) 
        for i in range(self.qt_inc_matrix.rowCount()):
            for j in range(self.qt_inc_matrix.columnCount()):
                self.qt_inc_matrix.setItem(i, j, QTableWidgetItem(str(0)))
        self.qt_inc_matrix.resizeColumnsToContents()	# делаем ресайз колонок по содержимому
        grid_layout.addWidget(self.qt_inc_matrix, 5, 0, 2, 1)   # Добавляем таблицу в сетку

        self.label_inc_tutorial = QLabel('Инструкции2:', self)
        grid_layout.addWidget(self.label_inc_tutorial, 5, 1, 1, 2)

        button_random_inc = QPushButton('Рандом', self) # кнопка заполнения матрицы inc рандомно
        grid_layout.addWidget(button_random_inc, 6, 1)
        button_random_inc.clicked.connect(self.button_random_inc_clicked)

        button_create_fr_inc = QPushButton('Создать граф', self) # создаём структуру графа и заполняем остальные таблицы автоматически
        grid_layout.addWidget(button_create_fr_inc, 6, 2)
        button_create_fr_inc.clicked.connect(self.button_create_fr_inc_clicked)

        # надпись "Информация о вершинах"
        self.label_nodes_info = QLabel('Информация о вершинах:', self)
        self.label_nodes_info.setMaximumHeight(30)
        grid_layout.addWidget(self.label_nodes_info, 7, 0)

        # таблица вершин графа
        self.qt_nodes_table = QTableWidget(self.nodes_count, 3, self)  # Создаём таблицу
        #self.qt_nodes_table.setMinimumSize(QSize(200, 200)) 
        self.qt_nodes_table.setMinimumHeight(150)
        '''for i in range(self.qt_nodes_table.rowCount()):
            for j in range(self.qt_nodes_table.columnCount()):
                self.qt_nodes_table.setItem(i, j, QTableWidgetItem(str(0)))
        self.qt_nodes_table.resizeColumnsToContents()	# делаем ресайз колонок по содержимому'''
        self.qt_nodes_table.setHorizontalHeaderLabels(["x", "y", "Связи"])  # Устанавливаем заголовки таблицы
        grid_layout.addWidget(self.qt_nodes_table, 8, 0, 2, 1)   # Добавляем таблицу в сетку

        self.label_nodes_inf_tutorial = QLabel('Инстр3:', self)
        grid_layout.addWidget(self.label_nodes_inf_tutorial, 8, 1)

        button_nodes_t_apply = QPushButton('Применить', self) # кнопка заполнения матрицы inc рандомно
        grid_layout.addWidget(button_nodes_t_apply, 9, 1, 1, 2)
        button_nodes_t_apply.clicked.connect(self.button_nodes_t_apply_clicked)

        button_draw_graph = QPushButton('Построить граф', self) # кнопка заполнения матрицы inc рандомно
        button_draw_graph.setStyleSheet('QPushButton {background-color: #4040ff; color: white;}')
        grid_layout.addWidget(button_draw_graph, 10, 0, 1, 2)
        button_draw_graph.clicked.connect(self.button_draw_graph_clicked)

        '''self.figure = plt.figure()  #figsize=(8, 8), dpi=80
        self.figure.set_size_inches(8, 8, forward=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Create the maptlotlib FigureCanvas object, 
        # which defines a single set of axes as self.axes.
        #self.graph = MplCanvas(self, width=5, height=4, dpi=100)
        #graph.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        grid_layout.addWidget(self.canvas, 0, 4, 5, 1)   # Добавляем таблицу в сетку
        grid_layout.addWidget(self.toolbar, 0, 4)   # Добавляем таблицу в сетку'''


    def button_matr_size_clicked(self):
        self.nodes_count = self.spinbox.value()
        self.qt_adj_matrix.setColumnCount(self.nodes_count)     # Устанавливаем колонки
        self.qt_adj_matrix.setRowCount(self.nodes_count)        # и строки таблице
        self.qt_inc_matrix.setRowCount(self.nodes_count)
        self.qt_nodes_table.setRowCount(self.nodes_count)

        # заполняем матрицу нулями
        for i in range(self.nodes_count):
            for j in range(self.nodes_count):
                self.qt_adj_matrix.setItem(i, j, QTableWidgetItem(str(0)))
        self.qt_adj_matrix.resizeColumnsToContents()	# делаем ресайз колонок по содержимому
        # заполняем матрицу нулями
        for i in range(self.nodes_count):
            for j in range(self.nodes_count):
                self.qt_inc_matrix.setItem(i, j, QTableWidgetItem(str(0)))
        self.qt_inc_matrix.resizeColumnsToContents()	# делаем ресайз колонок по содержимому

    def button_random_adj_clicked(self):
        #print(self.qt_adj_matrix.item(0,0).text())
        adj_array = np.random.choice([1, 0], size=self.nodes_count ** 2, p=[.5, .5])  #делаем одномерный массив, будущую матрицу, и заполняем её нужными значениями с указаниями вероятностей
        adj_matrix = np.reshape(adj_array, (self.nodes_count, self.nodes_count))    #матрица из массива
        np.fill_diagonal(adj_matrix, 0) # убираем петли
        #print(adj_matrix)
        for i in range(self.nodes_count):
            for j in range(self.nodes_count):
                self.qt_adj_matrix.setItem(i, j, QTableWidgetItem(str(adj_matrix[i][j])))
 
    def button_create_fr_adj_clicked(self):
        """создём граф и заполняем остальные таблицы по кнопке "создать граф"""
        # забираем матрицу из виджета
        adj_array  = np.array([], dtype=int)
        for i in range(self.nodes_count):
            for j in range(self.nodes_count):
                adj_array = np.append(adj_array, int(self.qt_adj_matrix.item(i,j).text()))
        self.adj_matrix = np.reshape(adj_array, (self.nodes_count, self.nodes_count))
        #print(adj_matrix)
        self.G = nx.DiGraph(self.adj_matrix)
        inc_matrix = -nx.incidence_matrix(self.G, oriented=True) # SciPy matrix
        inc_matrix = inc_matrix.toarray()   #numpy matrix
        inc_matrix = inc_matrix.astype(int) # делаем его int
        self.qt_inc_matrix.setColumnCount(inc_matrix.shape[1])     # Устанавливаем колонки
        for i in range(self.nodes_count):
            for j in range(inc_matrix.shape[1]):
                self.qt_inc_matrix.setItem(i, j, QTableWidgetItem(str(inc_matrix[i][j])))
        self.qt_inc_matrix.resizeColumnsToContents()	# делаем ресайз колонок по содержимому
        #pos = nx.kamada_kawai_layout(self.G)	# словарь позиций узлов
        for node in self.nodes_coords:
            #self.nodes_coords[node][0] = round(pos[node][0] * 1000, 2)    # делаем более приятные координаты, с округлением
            #self.nodes_coords[node][1] = abs(703 - self.nodes_coords[node][1])
            #self.pos = pos
            self.qt_nodes_table.setItem(node, 0, QTableWidgetItem(str(self.nodes_coords[node][0])))
            self.qt_nodes_table.setItem(node, 1, QTableWidgetItem(str(self.nodes_coords[node][1])))
    
    def button_random_inc_clicked(self):
        pass

    def button_create_fr_inc_clicked(self):
        pass

    def button_nodes_t_apply_clicked(self):
        pass
    
    def button_draw_graph_clicked(self):
        '''# create an axis
        #ax = self.figure.add_subplot(111)
        nx.draw_networkx(self.G, self.pos)    #with_labels=True
        axes = plt.gca()
        axes.set_xlim([-1000,1000])
        axes.set_ylim([-1000,1000])
        plt.savefig("C:\projects\python\Graphs_gui\Graph.png", format="PNG")
        #self.graph.plot()
        # refresh canvas
        self.canvas.draw()'''
        np.savetxt('export_file.txt', self.adj_matrix, fmt='%d')

        fig = plt.figure(figsize=(11.02, 7.03), dpi = 100, frameon=False)
        img = plt.imread("C:\projects\python\Ghraps_Lab_1\map.png")
        ax = fig.add_axes([0, 0, 1 ,1])
        ax.set_xlim([0,1102])
        ax.set_ylim([0,703])

        ax.imshow(img, extent=[0, 1102, 0, 703])
        nx.draw_networkx(self.G, self.nodes_coords, ax=ax)
        plt.show()
        
        
 
 
if __name__ == "__main__":
    import cv2
    import sys

    pos = {}    # позиции узлов графа, в дальнейшем передаются в конструктор
    nodes_count_user = 0
    # reading the image
    img = cv2.imread('map.png')
    height, width = img.shape[:2]
   
    # displaying the image
    cv2.imshow('image', img)    
    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', map_click_event)  
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    # close the window
    cv2.destroyAllWindows()

    app = QApplication(sys.argv)
    mw = MainWindow(nodes_count_user, pos)
    mw.show()
    sys.exit(app.exec())