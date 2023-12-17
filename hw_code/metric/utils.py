from math import isclose
import torch
from editdistance import eval


# rabotyaga dp implementation was here before...

#def calc_metric(target_list, predicted_list):
#  if len(target_list) == 0:
#    return 1
#  rows = len(target_list) + 1
#  cols = len(predicted_list) + 1
#
#  matrix = torch.zeros(rows, cols)
#  matrix[:, 0] = torch.arange(rows)
#  matrix[0, :] = torch.arange(cols)
#  for i in range(1, rows):
#    for j in range(1, cols):
#      if target_list[i - 1] == predicted_list[j - 1]:
#        matrix[i][j] = matrix[i - 1][j - 1]
#      else:
#        matrix[i][j] = min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]) + 1
#
#  return matrix[rows - 1][cols - 1] / (rows - 1)

def calc_metric(target_list, predicted_list):
  if len(target_list) == 0:
    return 1
  return eval(target_list, predicted_list) / len(target_list)


def calc_cer(target_text, predicted_text) -> float:
  #print('TARGET', target_text)
  #print('PREDICTED', predicted_text)
  return calc_metric(target_text, predicted_text)


def calc_wer(target_text, predicted_text) -> float:
  return calc_metric(target_text.split(), predicted_text.split())


def report_error(test_name, target_text, predicted_text, target_value, calculated_value):
  print(f"Failure in {test_name}()")
  print("Target Text:", target_text)
  print("Predicted Text:", predicted_text)
  print("Target Value:", target_value)
  print("Calculated Value:", calculated_value)
  assert False


def test_calc_cer():
  target_text = 'hello'
  predicted_text = 'helo'
  cer = calc_cer(target_text, predicted_text)
  if not isclose(cer, 0.2, abs_tol=1e-6):
    report_error("test_calc_cer", target_text, predicted_text, target_cer, our_cer)

  target_text = 'apple'
  predicted_text = 'ample'
  cer = calc_cer(target_text, predicted_text)
  if not isclose(cer, 0.2, abs_tol=1e-6):
    report_error("test_calc_cer", target_text, predicted_text, 0.2, cer)

  target_text = 'world'
  predicted_text = 'worllld'
  cer = calc_cer(target_text, predicted_text)
  if not isclose(cer, 0.4, abs_tol=1e-6):
    report_error("test_calc_cer", target_text, predicted_text, 0.4, cer)

  target_text = 'reference'
  predicted_text = 'reference'
  cer = calc_cer(target_text, predicted_text)
  if cer != 0:
    report_error("test_calc_cer", target_text, predicted_text, 0, cer)

  target_text = ''
  predicted_text = 'apple'
  cer = calc_cer(target_text, predicted_text)
  if not isclose(cer, 1, abs_tol=1e-6):
    report_error("test_calc_cer", target_text, predicted_text, 1, cer)


def test_calc_wer():
  target_text = 'hello world'
  predicted_text = 'helo world'
  wer = calc_wer(target_text, predicted_text)
  if not isclose(wer, 0.5, abs_tol=1e-6):
    report_error("test_calc_wer", target_text, predicted_text, 0.5, wer)

  target_text = 'apple orange'
  predicted_text = 'orange'
  wer = calc_wer(target_text, predicted_text)
  if not isclose(wer, 0.5, abs_tol=1e-6):
    report_error("test_calc_wer", target_text, predicted_text, 1, wer)

  target_text = 'orange'
  predicted_text = 'apple orange'
  wer = calc_wer(target_text, predicted_text)
  if not isclose(wer, 1, abs_tol=1e-6):
    report_error("test_calc_wer", target_text, predicted_text, 1, wer)

  target_text = 'the quick brown fox'
  predicted_text = 'brown quick the fox'
  wer = calc_wer(target_text, predicted_text)
  if not isclose(wer, 0.5, abs_tol=1e-6):
    report_error("test_calc_wer", target_text, predicted_text, 0.5, wer)

  target_text = 'the reference text'
  predicted_text = 'the reference text'
  wer = calc_wer(target_text, predicted_text)
  if wer != 0:
    report_error("test_calc_wer", target_text, predicted_text, 0, wer)


test_calc_cer()
test_calc_wer()
