"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])

    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """

    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.

    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    return np.random.choice(points, k)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!

    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.

    Parameters
    --------------------
        points         -- list of Points, dataset

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    initial_points = []
    return initial_points
    ### ========== TODO : END ========== ###


def compute_assignments_kMeans(clusters, points):
    """Computes assignments of points to clusters based on min dist.
    Params:
    clusters: list of points that represent clusters
    points: points to assign
    Returns:
    dict, where k: v = point: assigned cluster index
    """
    assigns = {}
    for p in points:
        min_dist_and_idx = min([(p.distance(clusters[i]),i)
                                for i in range(len(clusters))],
                               key = lambda x: x[0])
        idx = min_dist_and_idx[1]
        if idx in assigns:
            assigns[idx].append(p)
        else:
            assigns[idx] = [p]
    return assigns


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.

    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable:
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm

    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """

    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    cur_centroids = random_init(points, k) if init == 'random' else cheat_init(points, k)
    iters = 0
    prev_clusts = None
    while True:
        cur_cluster_set = ClusterSet([Cluster(v) for _, v
                                      in compute_assignments_kMeans(
                                          cur_centroids, points).items()])
        iters+=1
        print "iters: {}".format(iters)
        if prev_clusts is not None and cur_cluster_set.equivalent(prev_clusts):
            print "Done in {} iters".format(iters)
            return cur_cluster_set
        else:
            prev_clusts = cur_cluster_set
        if plot: plot_clusters(cur_cluster_set, 'Plot of kMeans Clusters', ClusterSet.centroids)
        new_centroids = cur_cluster_set.centroids()
        cur_centroids = new_centroids

    k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    cur_centroids = random_init(points, k) if init == 'random' else cheat_init(points, k)
    iters = 0
    prev_clusts = None
    while True:
        # compute assignments
        assignments = compute_assignments_kMeans(cur_centroids, points)
        clusts = [Cluster(v) for _, v in assignments.items()]
        cur_cluster_set = ClusterSet(clusts)
        iters+=1
        print "iters: {}".format(iters)
        if prev_clusts is not None and cur_cluster_set.equivalent(prev_clusts):
            print "Done in {} iters".format(iters)
            return cur_cluster_set
        else:
            prev_clusts = cur_cluster_set
        for c in clusts: assert(c in cur_cluster_set.members)
        if plot: plot_clusters(cur_cluster_set, 'Plot of KMediods Clusters', ClusterSet.medoids)
        new_centroids = cur_cluster_set.medoids()
        cur_centroids = new_centroids

    k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    X, y = get_lfw_data()
    mean_face = np.mean(X, axis = 0)
    U, mu = PCA(X)
    assert(np.sum(np.abs(mean_face - mu)) == 0)
    #show_image(vec_to_image(mu)) #PART A
    num_eigenfaces_to_plot = 12
    #plot_gallery([vec_to_image(U[:,i]) for i in xrange(num_eigenfaces_to_plot)]) #PART B
    for l in [1,10,50,100,500,1288]:
        Z, Ul = apply_PCA_from_Eig(X, U, l, mu)
        X_rec = reconstruct_from_PCA(Z, Ul, mu)
        #plot_gallery([vec_to_image(X_rec[i]) for i in xrange(num_eigenfaces_to_plot)]) #PART C
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part 2d-2f: cluster toy dataset
    print "generating data for clustering"
    np.random.seed(1234)
    pts = generate_points_2d(20)
    cluster_set = kMeans(pts, 3, plot = True)
    another_cluster_set = kMedoids(pts, 3, plot = True)
    exit()


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part 3a: cluster faces
    np.random.seed(1234)

    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)

    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    np.random.seed(1234)

    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
