------------------------------------------------------------------------
--[[ Clip ]]--
-- clips values within minval and maxval
------------------------------------------------------------------------
local Clip, parent = torch.class("nn.Clip", "nn.Module")

function Clip:__init(minval, maxval)
   assert(torch.type(minval) == 'number')
   assert(torch.type(maxval) == 'number')
   self.minval = minval
   self.maxval = maxval
   parent.__init(self)
end

function Clip:updateOutput(input)
   -- bound results within height and width
   self._mask = self._mask or input.new()
   self._byte = self._byte or torch.ByteTensor()
   
   self.output:resizeAs(input):copy(input)
   -- get the mask: greater than self.maxval will be 1 
   self._mask:gt(self.output, self.maxval)  -- if input is torch.CudaTensor, then mask will be also type of torch.CudaTensor
   local byte = torch.type(self.output) == 'torch.CudaTensor' and self._mask   -- if ouput = torch.CudaTensor, then byte = self._mask 
      or self._byte:resize(self._mask:size()):copy(self._mask)   -- if self.output is of type torch.DoubleTensor, then mask is of type torch.DoubleTensor 
   -- set elements greater than self.maxval to be self.maxval
   self.output[byte] = self.maxval   --torch.DoubleTensor cannot be indexed, so use torch.ByteTensor
   self._mask:lt(self.output, self.minval)
   byte = torch.type(self.output) == 'torch.CudaTensor' and self._mask 
      or self._byte:resize(self._mask:size()):copy(self._mask)
   -- set elements smaller than self.minval to be self.minval 
   self.output[byte] = self.minval
   return self.output
end

-- elements greater than maxval and elements less than minval, will not affect the the whole 
-- model's output, so gratient w.r.t this parameters will be zero. Thus if this model are inserted into the whole 
-- model, then it will not affecte the loss, this will be reflected in the gradOutput, which will already have 0 gradients 
function Clip:updateGradInput(input, gradOutput)
   self.gradInput:set(gradOutput)  -- self.gradInput just be set to gradOutput
   return self.gradInput
end

