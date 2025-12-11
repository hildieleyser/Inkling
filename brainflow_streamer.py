from brainflow import BoardShim, BrainFlowInputParams, BoardIds

params = BrainFlowInputParams()
params.serial_port = "/dev/tty.usbserial-DM00XXXXX"  # your Cyton port

board = BoardShim(BoardIds.CYTON_BOARD, params)
board.prepare_session()

board.start_stream()

print("Collecting...")
for _ in range(5):
    data = board.get_current_board_data(250)
    print(data.shape)

board.stop_stream()
board.release_session()
