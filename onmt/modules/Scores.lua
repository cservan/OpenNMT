--[[ nn unit. Maps from scores for each words
--]]
local Scores, parent = torch.class('onmt.Scores', 'nn.Container')

--[[
Parameters:

  * `scoresVec` - the scores for the embeddings
--]]
function Scores:__init(scoresVec)
  parent.__init(self)
  self.net = nn.Tensor(1,scoresVec:size())
  self.net = scoresVec:clone()
  self.net.gradWeight = nil
  self:add(self.net)
end

