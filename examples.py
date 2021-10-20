def RNNLstm():
    print('-------------------   RNN Lstm   --------------------')
    from data.load_data import load_coco_data, sample_coco_minibatch, decode_captions, image_from_url
    import matplotlib.pyplot as plt
    from layer import ActLayer, WeightLayer, EmbedLayer, ConnectEmbedWeightRNN, LstmRNN
    from model import RNN
    from activation import Softmax

    batch_size = m = 100
    data = load_coco_data(max_train=m)
    # for k in data:
    #     print(k, len(data[k]))
    hidden_dim = h = 512
    input_dim = d = data['train_features'].shape[1]
    data_dic = data['word_to_idx']
    timesteps = t = 16
    wordvec_dim = w = 256
    vocab_size = v = len(data_dic)
    lr, lrd = 2e-3, 0.995

    weight_layer = WeightLayer(h, input_shape=d, alpha=lr)
    embed_layer = EmbedLayer(w, input_shape=v, alpha=lr)
    connect_layer = ConnectEmbedWeightRNN(weight_layer=weight_layer, embed_layer=embed_layer)
    layers = [connect_layer, LstmRNN(alpha=lr), WeightLayer(v, alpha=lr), ActLayer(act=Softmax())]

    model = RNN(data_dic, layers)
    model.compile()
    data_train = (data['train_captions'][:m], data['train_features'][:m])
    val = (data['val_captions'][:m], data['val_features'][:m])

    model.load('lstm_coco')
    l1, l2 = model.train(data_train, val, epoch=1, return_loss=True, lrd=lrd)
    model.save('lstm_coco')

    plt.plot(range(len(l1)), l2)
    plt.legend(['t', 'v'])
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.title('Training Loss history')
    plt.show()

    # import metrics
    # print(metrics.print_metrics(y, model.predict(X)))
    # print(metrics.print_metrics(Yv, model.predict(Xv)))
    # print(metrics.print_metrics(Yte, model.predict(Xte)))

    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(data, split=split, batch_size=2)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])
        sample_captions = model.sample(features)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            img = image_from_url(url)
            # Skip missing URLs.
            if img is None:
                continue
            plt.imshow(img)
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.show()


RNNLstm()
