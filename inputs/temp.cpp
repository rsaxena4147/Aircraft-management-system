#include <bits/stdc++.h>
using namespace std;

int minRefuelStops(int target, int startFuel, vector<vector<int>> &stations)
{
  unordered_map<long long, long long> mp;
  for (auto s : stations)
  {
    mp[s[0]] = s[1];
  }
  long long maxReach = startFuel;
  long long count = 0;
  for (int i = 1; i <= target; i++)
  {
    // cout<<i<<" "<<startFuel<<" "<<maxReach<<endl;
    if (i > maxReach)
    {
      return -1;
    }
    if (i > startFuel)
    {
      count++;
      startFuel = maxReach;
    }
    if (mp.find(i) != mp.end())
      maxReach = max(maxReach, maxReach + mp[i]);
  }
  return count;
}
int main()
{
  vector<vector<int>> arr = {{5, 1000000000}, {1000, 1000000000}, {100000, 1000000000}};
  cout << minRefuelStops(1000000000, 1000000000 , arr );
}