local Bigrams, parent = torch.class("nn.Bigrams", "nn.Module")

--Function taken by torchx Aliasmultinomial.lua 
-- Alias method: sampling from discrete probability distribution 
function Bigrams:setup(probs)
   assert(probs:dim() == 1)   -- probs must be 1 dim 
   local K = probs:nElement() -- K is the number of probabilities 
   local q = probs.new(K):zero() -- q = [0, 0, ..., 0]: size K
   local J = torch.LongTensor(K):zero() -- J = [0, 0, ..., 0], size K

   -- Sort the data into the outcomes with probabilities
   -- that are larger and smaller than 1/K.
   local smaller, larger = {}, {} -- smaller stores income data with probabilities less than 1/K, vice versa for larger table 
   local maxk, maxp = 0, -1
   for kk = 1,K do
      local prob = probs[kk]
      q[kk] = K*prob  -- set q[kk] be K*probs[kk]
      if q[kk] < 1 then
         table.insert(smaller, kk) -- storing indices ranging from 1 to K 
      else
         table.insert(larger, kk)
      end
      if maxk > maxp then  -- maxk = 0, maxp = -1 
  
      end
   end
   
   -- Loop through and create little binary mixtures that
   -- appropriately allocate the larger outcomes over the
   -- overall uniform mixture.
   while #smaller > 0 and #larger > 0 do
      local small = table.remove(smaller) -- remove last element from table smaller 
      local large = table.remove(larger)

      J[small] = large  
      q[large] = q[large] - (1.0 - q[small])

      if q[large] < 1.0 then
         table.insert(smaller,large)
      else
         table.insert(larger,large)
      end
   end
   assert(q:min() >= 0)
   if q:max() > 1 then
      q:div(q:max())
   end
   assert(q:max() <= 1)
   if J:min() <= 0 then
      -- sometimes an large index isn't added to J. 
      -- fix it by making the probability 1 so that J isn't indexed.
      local i = 0
      J:apply(function(x)
         i = i + 1
         if x <= 0 then
            q[i] = 1
         end
      end)
   end
   return J, q
end


function Bigrams:batchdraw(output, J, q)
   assert(torch.type(output) == 'torch.LongTensor')
   assert(output:nElement() > 0)
   local K  = J:nElement()
   
   local _kk = output.new()
   _kk:resizeAs(output):random(1,K)
   
   local _q = q.new()
   _q:index(q, 1, _kk:view(-1))
   
   local _mask = torch.LongTensor()
   _mask:resize(_q:size()):bernoulli(_q)
   
   local __kk = output.new()
   __kk:resize(_kk:size()):copy(_kk)
   __kk:cmul(_mask)
   
   -- if mask == 0 then output[i] = J[kk[i]] else output[i] = 0
   
   _mask:add(-1):mul(-1) -- (1,0) - > (0,1)
   output:view(-1):index(J, 1, _kk:view(-1))
   output:cmul(_mask)
   
   -- elseif mask == 1 then output[i] = kk[i]
   
   output:add(__kk)
   return output
end


function Bigrams:__init(bigrams, nsample)
   self.nsample = nsample
   self.bigrams = bigrams
   self.q = {}
   self.J = {}
   for uniI, map in pairs(bigrams) do 
      local J, q = self.setup(self, map.prob)
      self.J[uniI] = J
      self.q[uniI] = q
   end   
end


function Bigrams:updateOutput(input)
   assert(torch.type(input) == 'torch.LongTensor')
   local batchSize = input:size(1)
   self.output = torch.type(self.output) == 'torch.LongTensor' and self.output or torch.LongTensor()
   self.output:resize(batchSize, self.nsample) 

   for i = 1, batchSize do
     self.batchdraw(self, self.output[i], self.J[input[i]], self.q[input[i]])
     self.output[i]:apply(function(x) return self.bigrams[input[i]]['index'][x] end)
   end
   
   return self.output   
end

function Bigrams:updateGradInput(input, gradOutput)
   self.gradInput = torch.type(self.gradInput) == 'torch.LongTensor' or torch.LongTensor()
   self.gradInput:resizeAs(input):fill(0) 
   return self.gradInput
end

function Bigrams:statistics()
   local sum, count = 0, 0
   for uniI, map in pairs(self.bigrams) do 
      sum = sum + map.prob:nElement()
      count = count + 1
   end
   local meansize = sum/count
   return meansize
end
