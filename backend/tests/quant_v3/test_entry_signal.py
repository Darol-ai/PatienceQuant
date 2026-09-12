from app.quant_v3.entry_signal import entry_approved


def test_entry_approved_when_both_thresholds_comfortably_met():
    assert entry_approved(p_up=0.65, p_down=0.20) is True


def test_entry_rejected_when_p_up_too_low():
    assert entry_approved(p_up=0.55, p_down=0.20) is False


def test_entry_rejected_when_p_down_too_high():
    assert entry_approved(p_up=0.65, p_down=0.30) is False


def test_entry_boundaries_are_inclusive():
    """V3 方案第 4 节表格：p_up>=0.60 且 p_down<=0.25，边界含等号。"""
    assert entry_approved(p_up=0.60, p_down=0.25) is True


def test_entry_approved_accepts_overridden_thresholds():
    """门槛可覆盖——用于"训练集分位数代替原文绝对数字"这条改动
    （见 docs/adr），默认值不变，不传就还是 V3 原文的 0.60/0.25。"""
    assert entry_approved(p_up=0.19, p_down=0.21, p_up_threshold=0.19, p_down_threshold=0.21) is True
    assert entry_approved(p_up=0.19, p_down=0.21, p_up_threshold=0.60, p_down_threshold=0.25) is False
