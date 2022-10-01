from pytorch_tao.plugins import events
from ignite.engine import Events, CallableEventWithFilter


def test_events_register():

    @events.on(Events.COMPLETED)
    def func():
        pass

    assert func._tao_event == Events.COMPLETED


def test_events_filter_every():

    @events.on(Events.EPOCH_COMPLETED(every=3))
    def func():
        pass

    assert isinstance(func._tao_event, CallableEventWithFilter)
    assert not func._tao_event.filter(None, 2)
    assert func._tao_event.filter(None, 3)


def test_events_filter_once():

    @events.on(Events.EPOCH_COMPLETED(once=10))
    def func():
        pass

    assert isinstance(func._tao_event, CallableEventWithFilter)
    assert not func._tao_event.filter(None, 2)
    assert func._tao_event.filter(None, 10)
