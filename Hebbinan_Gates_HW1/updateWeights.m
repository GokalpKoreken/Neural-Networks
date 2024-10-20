function updateWeights(G, h, w1, w2, B)
   newLabels = {sprintf('w1=%d', w1), sprintf('w2=%d', w2), sprintf('B=%d', B), 'SUM', 'OUTPUT'};
    h.NodeLabel = newLabels;
end
