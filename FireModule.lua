--[[
  Fire module as explained in SqueezeNet http://arxiv.org/pdf/1602.07360v1.pdf.
--]]
--FIXME works only for batches.

local FireModule, Parent = torch.class('nn.FireModule', 'nn.Decorator')

function FireModule:__init(nInputPlane, s1x1, e1x1, e3x3, activation)
   self.nInputPlane = nInputPlane
   -- squeeze module
   self.s1x1 = s1x1 -- the number of convolutional filters(all will be 1x1) in sequeeze module
   -- expand module 
   self.e1x1 = e1x1 -- the number of 1x1 filters in expand module
   self.e3x3 = e3x3 -- the number of 3x3 filters in expand module
   self.activation = activation or 'ReLU'

   if self.s1x1 > (self.e1x1 + self.e3x3) then -- to limit #input channels to 3x3 filters
      print('Warning: <FireModule> s1x1 is recommended to be smaller'..
            ' then e1x1+e3x3')
   end
   
   self.module = nn.Sequential()
  -- squeeze module 
  -- nInputPlane, s1x1 output channel, 1 x 1 filters
   self.squeeze = nn.SpatialConvolution(nInputPlane, s1x1, 1, 1)
   -- expand module
   self.expand = nn.Concat(2) -- concatenate along second dimension 
   -- output: bz x e1x1 x H x W
   self.expand:add(nn.SpatialConvolution(s1x1, e1x1, 1, 1)) 
   -- output: bz x e3x3 x H x W
   self.expand:add(nn.SpatialConvolution(s1x1, e3x3, 3, 3, 1, 1, 1, 1))
    -- output of self.expand: bz x (e1x1+e3x3) x H x W  
  
   -- Fire Module
   self.module:add(self.squeeze)
   self.module:add(nn[self.activation]())
   self.module:add(self.expand)
   self.module:add(nn[self.activation]())
   
   Parent.__init(self, self.module)
end

--[[
function FireModule:type(type, tensorCache)
   assert(type, 'Module: must provide a type to convert to')
   self.module = nn.utils.recursiveType(self.module, type, tensorCache)
end
--]]

function FireModule:__tostring__()
   return string.format('%s inputPlanes: %d -> Squeeze Planes: %d -> '..
                        'Expand: %d(1x1) + %d(3x3), activation: %s',
                        torch.type(self), self.nInputPlane, self.s1x1,
                        self.e1x1, self.e3x3, self.activation)
end
