------------------------------------------------------------------------
--[[ Constant ]]--
-- Outputs a constant value given an input.
-- If nInputDim is specified, uses the input to determine the size of 
-- the batch. The value is then replicated over the batch.
-- You can use this with nn.ConcatTable() to append constant inputs to
-- an input : nn.ConcatTable():add(nn.Constant(v)):add(nn.Identity()) .
------------------------------------------------------------------------

-- this module has no trainable parameters 
-- useful when you want to output a value that 
-- is independent of the input to the neural network
local Constant, parent = torch.class("nn.Constant", "nn.Module")

-- value: torch.Tensor 
function Constant:__init(value, nInputDim)
   self.value = value
   if torch.type(self.value) == 'number' then  -- if value is a number, convert it to a torch.Tensor object
      self.value = torch.Tensor{self.value}
   end
   assert(torch.isTensor(self.value), "Expecting number or tensor at arg 1")
   self.nInputDim = nInputDim -- set self.nInputDim to nInputDim to be nInputDim
   parent.__init(self)
end

function Constant:updateOutput(input)
   if self.nInputDim and input:dim() > self.nInputDim then -- input has batch dim 
      local vsize = self.value:size():totable() 
      self.output:resize(input:size(1), table.unpack(vsize))  -- self.output: bz x table.unpack(vsize)
      local value = self.value:view(1, table.unpack(vsize))  
      self.output:copy(value:expand(self.output:size()))  -- expand value to be: bz x table.unpack(vsize) 
   else -- input doesnot have  batch dim 
      self.output:resize(self.value:size()):copy(self.value)  -- just output self.value
   end
   return self.output
end

function Constant:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()  -- no matter what the input, the output will be a const, so here self.gradInput:zero()
   return self.gradInput
end
