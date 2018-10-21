# import pickle
import knn
import imtools
from knn import *
import numpy as np

def classify(x,y,model=model):
    return np.array([model.classify([xx,yy]) for (xx,yy) in zip(x,y)])

def main():
    # with open(’points_normal.pkl’, ’r’) as f:
    #         class_1 = pickle.load(f)
    #         class_2 = pickle.load(f)
    #         labels = pickle.load(f)

    model = knn.KnnClassifier(labels,vstack((class_1,class_2)))

    # with open(’points_normal_test.pkl’, ’r’) as f:
    #     class_1 = pickle.load(f)
    #     class_2 = pickle.load(f)
    #     labels = pickle.load(f)

    print(model.classify(class_1[0]))

    imtools.plot_2D_boundary([-6,6,-6,6],[class_1,class_2],classify,[1,-1])
    show()

def plot_2D_boundary(plot_range,points,decisionfcn,labels,values=[0]):
    """ Plot_range is (xmin,xmax,ymin,ymax), points is a list
    of class points, decisionfcn is a funtion to evaluate,
    labels is a list of labels that decisionfcn returns for each class,
    values is a list of decision contours to show. """

    clist = [’b’,’r’,’g’,’k’,’m’,’y’] # cores das classes

    # evaluate on a grid and plot contour of decision function
    x = arange(plot_range[0],plot_range[1],.1)
    y = arange(plot_range[2],plot_range[3],.1)
    xx,yy = meshgrid(x,y)
    xxx,yyy = xx.flatten(),yy.flatten() # lists of x,y in grid
    zz = array(decisionfcn(xxx,yyy))
    zz = zz.reshape(xx.shape)

    contour(xx,yy,zz,values)

    for i in range(len(points)):
        d = decisionfcn(points[i][:,0],points[i][:,1])
        correct_ndx = labels[i]==d
        incorrect_ndx = labels[i]!=d
        plot(points[i][correct_ndx,0],points[i][correct_ndx,1],’*’,color=clist[i])
        plot(points[i][incorrect_ndx,0],points[i][incorrect_ndx,1],’o’,color=clist[i])

    axis(’equal’)

if __name__ == '__main__':
    main()