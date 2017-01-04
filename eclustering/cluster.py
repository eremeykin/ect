__author__ = 'eremeykin'


class Cluster(object):

    def __init__(self, content, centroid):
        self.content = content
        self.centroid = centroid
        self.size = len(content)

    def __str__(self):
        return "(l=" + str(len(self.content)) + " c=" + str(self.centroid) + ")"

    def __repr__(self):
        return str(self)
