import copy


def initialize_ema_model(model):
    """Create a frozen EMA teacher from the student model."""
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def update_ema(teacher, student, decay=0.99):
    """EMA update: teacher = decay * teacher + (1 - decay) * student."""
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)
