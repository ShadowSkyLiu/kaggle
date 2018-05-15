from mxnet import nd, gluon
import mxnet as mx
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iter, mx.io.MXDataIter):
        data_iter.reset()
    for batch in data_iter:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            y = y.astype('float32')
            acc += nd.sum(net(X).argmax(axis=1) == y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()  # don't push too many operators into backend
    return acc.asscalar() / n

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
