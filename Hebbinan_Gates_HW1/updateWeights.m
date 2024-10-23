function updateWeights(G, h, w1, w2, B)
    edgeLabels = cell(1, G.numedges);
    edgeLabels{1} = sprintf('w1=%.2f', w1);
    edgeLabels{2} = sprintf('w2=%.2f', w2);
    edgeLabels{3} = sprintf('B=%.2f', B);
    edgeLabels{4} = '';
    
    h.EdgeLabel = edgeLabels;
end
