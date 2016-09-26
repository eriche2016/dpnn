
local CAddTensorTable, parent = torch.class('nn.CAddTensorTable', 'nn.Module')

function CAddTensorTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

-- input is a table with 2 entries. input[1] is the vector to be added.
-- input[2] is the table to which we add the vector
-- input = {A, {B1, B2, B3, ..., B_m}}
-- output = {A+B1, A+B2, ..., A+B_m}
function CAddTensorTable:updateOutput(input)
  local currentOutput = {}
  for i=1,#input[2] do
    currentOutput[i] = currentOutput[i] or input[1].new()
    currentOutput[i]:resizeAs(input[1])
    currentOutput[i]:copy(input[2][i])
    currentOutput[i]:add(input[1])
  end
  -- overwrite all the #input[2]+1,... ,#currentOutput to nil
  for i = #input[2]+1, #currentOutput do
    currentOutput[i] = nil
  end
   -- self.output
  self.output = currentOutput
  return self.output
end

-- input = {A, {B1, B2, B3, ..., B_m}}
function CAddTensorTable:updateGradInput(input, gradOutput)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[1]:resizeAs(input[1])
   
   -- accumulates gradient with A: gradA
  self.gradInput[1]:copy(gradOutput[1]) 
  for i=2, #input[2] do
    self.gradInput[1]:add(gradOutput[i])
  end
   
  -- gradient with B_i, a table: {gradB1, gradB2, ..., gradB_m}
  self.gradInput[2] = self.gradInput[2] or {}
  for i=1,#input[2] do
    self.gradInput[2][i] = self.gradInput[2][i] or input[1].new()
    self.gradInput[2][i]:resizeAs(input[1])
    self.gradInput[2][i]:copy(gradOutput[i])
  end
   
  for i=#input[2]+1, #self.gradInput[2] do
     self.gradInput[2][i] = nil
  end
  -- {gradA, {gradB1, gradB2, ..., gradB_m}}
  return self.gradInput
end
