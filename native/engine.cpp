// ScratchChess: original board, move generation, incremental network and search.
// No Stockfish code or evaluation parameters are used here.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
using U64=uint64_t;
constexpr int MAXPLY=120, MAXW=512, MAXH=128, INF=32000, MATE=30000;
constexpr const char* START="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
U64 Z[13][64], Zcastle[16], Zep[8], Zside;
U64 rngstate=0x94d049bb133111ebULL;
U64 random64(){U64 z=(rngstate+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
int side(int p){return p<0;} int sign(int c){return c?-1:1;}
struct Move {int from=0,to=0,promo=0; bool operator==(const Move& b)const{return from==b.from&&to==b.to&&promo==b.promo;} bool valid()const{return from!=to;} };
std::string uci(Move m){if(!m.valid())return "0000";std::string s;s+=char('a'+m.from%8);s+=char('1'+m.from/8);s+=char('a'+m.to%8);s+=char('1'+m.to/8);if(m.promo)s+="  nbrqk"[m.promo];return s;}
struct List {std::array<Move,256> v;int n=0;void add(int f,int t,int p=0){v[n++]={f,t,p};}};
struct Board {
    std::array<int8_t,64> sq{};int turn=0,castle=0,ep=-1,half=0,full=1;int king[2]={4,60};U64 key=0;
    void set(int s,int p){int old=sq[s];if(old)key^=Z[old+6][s];sq[s]=p;if(p){key^=Z[p+6][s];if(abs(p)==6)king[side(p)]=s;}}
    bool attacked(int s,int by)const {
        int x=s%8,y=s/8,sg=sign(by),py=y-sg;
        if(py>=0&&py<8)for(int dx:{-1,1})if(x+dx>=0&&x+dx<8&&sq[py*8+x+dx]==sg)return true;
        static const int nx[8]={1,2,2,1,-1,-2,-2,-1},ny[8]={2,1,-1,-2,-2,-1,1,2};
        for(int i=0;i<8;i++){int a=x+nx[i],b=y+ny[i];if(a>=0&&a<8&&b>=0&&b<8&&sq[b*8+a]==sg*2)return true;}
        static const int dx[8]={1,-1,0,0,1,1,-1,-1},dy[8]={0,0,1,-1,1,-1,1,-1};
        for(int d=0;d<8;d++)for(int n=1,a=x+dx[d],b=y+dy[d];a>=0&&a<8&&b>=0&&b<8;a+=dx[d],b+=dy[d],n++){
            int p=sq[b*8+a];if(!p)continue;
            if(side(p)==by&&(abs(p)==5||(d<4&&abs(p)==4)||(d>=4&&abs(p)==3)||(n==1&&abs(p)==6)))return true;
            break;
        }return false;
    }
    bool check()const{return attacked(king[turn],turn^1);}
    void fen(const std::string& text){*this=Board();std::istringstream in(text);std::string cells,t,c,e;in>>cells>>t>>c>>e;if(!(in>>half>>full)){half=0;full=1;}int r=7,f=0;std::string names=" PNBRQK";
        for(char ch:cells){if(ch=='/'){r--;f=0;}else if(ch>='1'&&ch<='8')f+=ch-'0';else {bool black=ch>='a'&&ch<='z';char upper=black?ch-32:ch;size_t p=names.find(upper);if(p==std::string::npos||r<0||f>=8)throw std::runtime_error("bad FEN");set(r*8+f++,sign(black)*int(p));}}
        turn=t=="b";castle=(c.find('K')!=c.npos?1:0)|(c.find('Q')!=c.npos?2:0)|(c.find('k')!=c.npos?4:0)|(c.find('q')!=c.npos?8:0);
        ep=e=="-"?-1:(e[1]-'1')*8+e[0]-'a'; key^=Zcastle[castle];if(turn)key^=Zside;if(ep>=0)key^=Zep[ep%8];
    }
    Board moved(Move m)const {
        Board b=*this;int p=sq[m.from],capt=sq[m.to];
        b.key^=Zcastle[b.castle];if(b.ep>=0)b.key^=Zep[b.ep%8];b.ep=-1;
        b.half++;if(abs(p)==1||capt)b.half=0;
        b.set(m.from,0);b.set(m.to,m.promo?sign(turn)*m.promo:p);
        if(abs(p)==1&&m.to==ep&&!capt){b.set(m.to-sign(turn)*8,0);b.half=0;}
        if(abs(p)==1&&abs(m.to-m.from)==16)b.ep=(m.from+m.to)/2;
        if(abs(p)==6){b.castle&=turn?3:12;if(abs(m.to-m.from)==2){int rook=m.to>m.from?m.from+3:m.from-4;b.set(rook,0);b.set((m.from+m.to)/2,sign(turn)*4);}}
        for(int s:{m.from,m.to}){if(s==0)b.castle&=~2;if(s==7)b.castle&=~1;if(s==56)b.castle&=~8;if(s==63)b.castle&=~4;}
        b.turn^=1;if(turn)b.full++;b.key^=Zside;b.key^=Zcastle[b.castle];if(b.ep>=0)b.key^=Zep[b.ep%8];return b;
    }
    Board nullmove()const{Board b=*this;if(b.ep>=0)b.key^=Zep[b.ep%8];b.ep=-1;b.turn^=1;b.key^=Zside;b.half++;return b;}
    bool legalep()const{if(ep<0)return false;for(int dx:{-1,1}){int from=ep-sign(turn)*8+dx;if(from>=0&&from<64&&abs(from%8-ep%8)==1&&sq[from]==sign(turn)){Board b=moved({from,ep,0});if(!b.attacked(b.king[turn],turn^1))return true;}}return false;}
    U64 repkey()const{return ep>=0&&!legalep()?key^Zep[ep%8]:key;}
    List pseudo(bool tactical=false)const {
        List out;int sg=sign(turn);
        auto pawn=[&](int a,int b){if(b/8==0||b/8==7){for(int p:{5,4,3,2})out.add(a,b,p);}else out.add(a,b);};
        static const int nx[8]={1,2,2,1,-1,-2,-2,-1},ny[8]={2,1,-1,-2,-2,-1,1,2};
        static const int dx[8]={1,-1,0,0,1,1,-1,-1},dy[8]={0,0,1,-1,1,-1,1,-1};
        for(int s=0;s<64;s++){int p=sq[s];if(!p||side(p)!=turn)continue;int type=abs(p),x=s%8,y=s/8;
            if(type==1){int t=s+sg*8;if(t>=0&&t<64&&!sq[t]){if(!tactical||t/8==0||t/8==7)pawn(s,t);int t2=s+sg*16;if(!tactical&&y==(turn?6:1)&&!sq[t2])out.add(s,t2);}
                for(int df:{-1,1})if(x+df>=0&&x+df<8){t=s+sg*8+df;if(t>=0&&t<64&&((sq[t]&&side(sq[t])!=turn&&abs(sq[t])!=6)||t==ep))pawn(s,t);}continue;}
            if(type==2){for(int i=0;i<8;i++){int a=x+nx[i],b=y+ny[i];if(a<0||a>7||b<0||b>7)continue;int t=b*8+a,q=sq[t];if(q&&(side(q)==turn||abs(q)==6))continue;if(!tactical||q)out.add(s,t);}continue;}
            for(int d=0;d<8;d++){if((type==3&&d<4)||(type==4&&d>=4))continue;for(int a=x+dx[d],b=y+dy[d];a>=0&&a<8&&b>=0&&b<8;a+=dx[d],b+=dy[d]){int t=b*8+a,q=sq[t];if(q&&side(q)==turn)break;if(abs(q)==6)break;if(!tactical||q)out.add(s,t);if(q||type==6)break;}}
            if(type==6&&!tactical&&s==(turn?60:4)&&!attacked(s,turn^1)){
                int k=turn?4:1,q=turn?8:2;
                if((castle&k)&&sq[s+3]==sg*4&&!sq[s+1]&&!sq[s+2]&&!attacked(s+1,turn^1)&&!attacked(s+2,turn^1))out.add(s,s+2);
                if((castle&q)&&sq[s-4]==sg*4&&!sq[s-1]&&!sq[s-2]&&!sq[s-3]&&!attacked(s-1,turn^1)&&!attacked(s-2,turn^1))out.add(s,s-2);
            }
        }return out;
    }
    List legal(bool tactical=false)const{List result,all=pseudo(tactical);for(int i=0;i<all.n;i++){Board b=moved(all.v[i]);if(!b.attacked(b.king[turn],turn^1))result.v[result.n++]=all.v[i];}return result;}
    bool anylegal()const{List all=pseudo();for(int i=0;i<all.n;i++){Board b=moved(all.v[i]);if(!b.attacked(b.king[turn],turn^1))return true;}return false;}
    bool insufficient()const{int minors=0,bishops=0,color=-1;for(int s=0;s<64;s++){int p=abs(sq[s]);if(p==1||p==4||p==5)return false;if(p==2)minors++;if(p==3){bishops++;int c=(s%8+s/8)%2;if(color==-1)color=c;else if(color!=c)color=2;}}return minors+bishops<=1||(minors==0&&color<2);}
};
struct Acc {alignas(32) float a[2][MAXW];};
struct Network {
    int w=0,h=0;std::vector<float> embedding,bias,fc1,b1,fc2,b2,out;float bo=0;
    void load(const std::string& path){std::ifstream f(path,std::ios::binary);char magic[8];uint32_t v,W,H;f.read(magic,8);f.read((char*)&v,4);f.read((char*)&W,4);f.read((char*)&H,4);if(!f||std::string(magic,8)!="SCRATCH1"||v!=1||W>MAXW||H>MAXH||W<1||H<1)throw std::runtime_error("invalid network");w=W;h=H;
        auto read=[&](std::vector<float>&a,size_t n){a.resize(n);f.read((char*)a.data(),n*4);};read(embedding,12289*w);read(bias,w);read(fc1,h*(2*w+12));read(b1,h);read(fc2,h*h);read(b2,h);read(out,h);f.read((char*)&bo,4);if(!f)throw std::runtime_error("truncated network");}
    int bucket(const Board&b,int c)const{int k=b.king[c]^(c?56:0);return(k/8/2)*4+(k%8/2);}
    int feature(int p,int s,int c,int k)const{return k*768+(abs(p)-1+(side(p)==c?0:6))*64+(s^(c?56:0));}
    // Tight, non-aliased row loops so the compiler emits SIMD. A per-element branch (or a
    // possibly-aliased destination) kept these scalar and made updates ~20x slower.
    static float dot(const float* __restrict a,const float* __restrict b,int n){float s=0;for(int j=0;j<n;j++)s+=a[j]*b[j];return s;}
    static void addRow(float* __restrict dst,const float* __restrict row,int n){for(int j=0;j<n;j++)dst[j]+=row[j];}
    static void subRow(float* __restrict dst,const float* __restrict row,int n){for(int j=0;j<n;j++)dst[j]-=row[j];}
    void rebuild(const Board&b,Acc&a,int c)const{std::copy(bias.begin(),bias.end(),a.a[c]);int k=bucket(b,c);for(int s=0;s<64;s++)if(b.sq[s])addRow(a.a[c],&embedding[size_t(feature(b.sq[s],s,c,k))*w],w);}
    void init(const Board&b,Acc&a)const{rebuild(b,a,0);rebuild(b,a,1);}
    void update(const Board&before,const Board&after,const Acc&src,Acc&dst,Move m)const{
        int changed[6]={m.from,m.to,-1,-1,-1,-1};int n=2,p=before.sq[m.from];
        if(abs(p)==1&&m.to==before.ep&&!before.sq[m.to])changed[n++]=m.to-sign(before.turn)*8;
        if(abs(p)==6&&abs(m.to-m.from)==2){changed[n++]=m.to>m.from?m.from+3:m.from-4;changed[n++]=(m.from+m.to)/2;}
        for(int c=0;c<2;c++){int k=bucket(after,c);if(k!=bucket(before,c)){rebuild(after,dst,c);continue;}std::copy(src.a[c],src.a[c]+w,dst.a[c]);
            for(int i=0;i<n;i++){int s=changed[i],old=before.sq[s],now=after.sq[s];if(old==now)continue;
                if(old)subRow(dst.a[c],&embedding[size_t(feature(old,s,c,k))*w],w);
                if(now)addRow(dst.a[c],&embedding[size_t(feature(now,s,c,k))*w],w);
            }
        }
    }
    float value(const Board&b,const Acc&a)const{
        alignas(32) float x[2*MAXW+12],y[MAXH],z[MAXH];int size=2*w+12;
        for(int c=0;c<2;c++)for(int i=0;i<w;i++)x[c*w+i]=std::clamp(a.a[b.turn^c][i],0.0f,1.0f);
        for(int i=0;i<12;i++)x[2*w+i]=0;
        for(int c=0;c<2;c++){int col=b.turn^c;x[2*w+c*2]=bool(b.castle&(col?4:1));x[2*w+c*2+1]=bool(b.castle&(col?8:2));}
        if(b.legalep())x[2*w+4+b.ep%8]=1;
        for(int i=0;i<h;i++)y[i]=std::clamp(b1[i]+dot(&fc1[size_t(i)*size],x,size),0.0f,1.0f);
        for(int i=0;i<h;i++){float s=b2[i];for(int j=0;j<h;j++)s+=fc2[i*h+j]*y[j];z[i]=std::clamp(s,0.0f,1.0f);}
        float s=bo;for(int i=0;i<h;i++)s+=out[i]*z[i];return std::tanh(s);
    }
    int eval(const Board&b,const Acc&a)const{return int(400*std::atanh(std::clamp(value(b,a),-0.99999f,0.99999f)));}
};
struct Entry {U64 key=0;Move move;int score=0,depth=-1,flag=0,half=0;};
struct Search {
    Network& net;std::vector<Entry> tt;std::array<Acc,MAXPLY+2> acc;
    std::vector<U64> past,path;Move killers[MAXPLY][2];int history[2][64][64]{};
    std::atomic<bool> stop{false};U64 nodes=0,nodeLimit=~U64(0);double milliseconds=1000,softMilliseconds=1000;std::chrono::steady_clock::time_point start;Move rootBest;
    bool pruning=true;int nullPly=-1;
    Search(Network&n):net(n),tt(1<<20){}
    double elapsed()const{return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
    void tick(){nodes++;if(stop||nodes>nodeLimit||((nodes&255)==0&&elapsed()>=milliseconds))throw 1;}
    int repeat(const Board&b,int ply)const{if(nullPly>=0)return 0;U64 key=b.repkey();int n=0;
        for(int i=int(path.size())-3;i>=0&&int(path.size())-1-i<=b.half;i-=2)if(path[i]==key)return 2; // cycle INSIDE search
        int remain=b.half-ply;for(int i=int(past.size())-2;i>=0&&int(past.size())-1-i<=remain;i--)if(past[i]==key)n++;
        return n>=2?3:0;
    }
    int normalize(int s,int ply)const{return s>MATE-MAXPLY?s+ply:s<-MATE+MAXPLY?s-ply:s;}
    int denormalize(int s,int ply)const{return s>MATE-MAXPLY?s-ply:s<-MATE+MAXPLY?s+ply:s;}
    int order(const Board&b,Move m,Move preferred,int ply)const{if(m==preferred)return 2000000;int values[7]={0,100,320,330,500,900,20000};int victim=abs(b.sq[m.to]);if(!victim&&abs(b.sq[m.from])==1&&m.to==b.ep)victim=1;
        if(victim)return 1000000+16*values[victim]-values[abs(b.sq[m.from])]+values[m.promo];if(m.promo)return 900000+values[m.promo];if(m==killers[ply][0])return 800000;if(m==killers[ply][1])return 700000;return history[b.turn][m.from][m.to];}
    void sort(List&list,const Board&b,Move best,int ply){std::array<int,256> scores;for(int i=0;i<list.n;i++)scores[i]=order(b,list.v[i],best,ply);for(int i=1;i<list.n;i++){Move m=list.v[i];int s=scores[i],j=i;while(j>0&&scores[j-1]<s){list.v[j]=list.v[j-1];scores[j]=scores[j-1];j--;}list.v[j]=m;scores[j]=s;}}
    int qsearch(const Board&b,int alpha,int beta,int ply,int qdepth=0){tick();bool check=b.check();if(ply>=MAXPLY-1)return net.eval(b,acc[ply]);List moves=b.legal(!check);
        // Stalemate is detected by the main search; probing every quiet leaf costs a full move generation.
        if(!moves.n&&check)return -MATE+ply;
        if(b.insufficient()||(ply>0&&(b.half>=100||repeat(b,ply))))return 0;
        int stand=-INF;
        if(!check){stand=net.eval(b,acc[ply]);if(stand>=beta)return stand;alpha=std::max(alpha,stand);if(qdepth>=16)return alpha;}
        sort(moves,b,{},ply);
        static const int gain[7]={0,100,320,330,500,900,0};
        for(int i=0;i<moves.n;i++){Move m=moves.v[i];
            // Delta pruning: even winning this piece (plus a margin) can't lift us to alpha.
            if(!check&&!m.promo&&alpha>-MATE+MAXPLY){int victim=b.sq[m.to]?abs(b.sq[m.to]):1;if(stand+gain[victim]+200<=alpha)continue;}
            Board child=b.moved(m);net.update(b,child,acc[ply],acc[ply+1],m);path.push_back(child.repkey());int s;
            try{s=-qsearch(child,-beta,-alpha,ply+1,qdepth+1);}catch(...){path.pop_back();throw;}path.pop_back();if(s>=beta)return s;alpha=std::max(alpha,s);
        }return alpha;
    }
    int search(const Board&b,int depth,int alpha,int beta,int ply,bool allowNull=true,int extensions=0){
        if(depth<=0)return qsearch(b,alpha,beta,ply);tick();if(ply>=MAXPLY-1)return net.eval(b,acc[ply]);bool check=b.check();
        List moves=b.legal();if(!moves.n)return check?-MATE+ply:0;
        if(b.insufficient())return 0;
        if(ply>0&&(b.half>=100||repeat(b,ply)))return 0;
        if(check&&extensions<8){depth++;extensions++;}
        int oldAlpha=alpha;U64 key=b.repkey();Entry &slot=tt[key&(tt.size()-1)];Move preferred=slot.key==key?slot.move:Move{};
        bool pv=beta-alpha>1;
        // Score cutoffs at non-PV nodes only, as in mainstream engines. Repetition-dependent
        // scores can leak between histories (graph history interaction); the PV stays exact
        // and positions near the 50-move limit are never cut.
        if(!pv&&ply>0&&b.half<90&&slot.key==key&&slot.depth>=depth){int s=denormalize(slot.score,ply);if(slot.flag==0||(slot.flag==1&&s>=beta)||(slot.flag==2&&s<=alpha))return s;}
        int staticEval=check?-INF:net.eval(b,acc[ply]);
        // Reverse futility: far above beta at shallow depth, a quiet move is very unlikely to fall below it.
        if(pruning&&!pv&&!check&&ply>0&&depth<=6&&abs(beta)<MATE-MAXPLY&&staticEval-90*depth>=beta)return staticEval;
        if(pruning&&allowNull&&!pv&&!check&&depth>=3&&ply>0&&b.half<90){bool material=false;for(int s=0;s<64;s++)if(b.sq[s]&&side(b.sq[s])==b.turn&&abs(b.sq[s])>=2&&abs(b.sq[s])<=5)material=true;
            if(material&&staticEval>=beta){Board child=b.nullmove();acc[ply+1]=acc[ply];int oldNull=nullPly;nullPly=ply;path.push_back(child.repkey());int s;
                try{s=-search(child,depth-3-depth/6,-beta,-beta+1,ply+1,false,extensions);}catch(...){path.pop_back();nullPly=oldNull;throw;}path.pop_back();nullPly=oldNull;if(s>=beta&&abs(s)<MATE-MAXPLY)return s;
            }
        }
        sort(moves,b,preferred,ply);int best=-INF;Move bestMove;
        // Internal iterative reduction: without a TT move this node is likely unimportant or new.
        if(pruning&&depth>=4&&!preferred.valid()&&!check)depth--;
        int quietsSeen=0,triedCount=0;Move tried[64];
        for(int i=0;i<moves.n;i++){Move m=moves.v[i];bool quiet=!b.sq[m.to]&&!(abs(b.sq[m.from])==1&&m.to==b.ep)&&!m.promo;Board child=b.moved(m);
            if(quiet)quietsSeen++;
            // Shallow quiet-move pruning once a move has been searched: late move pruning and futility.
            if(pruning&&!pv&&!check&&quiet&&i>0&&best>-MATE+MAXPLY&&!child.check()){
                if(depth<=4&&quietsSeen>3+depth*depth)continue;
                if(depth<=3&&staticEval+120*depth+80<=alpha)continue;
            }
            net.update(b,child,acc[ply],acc[ply+1],m);path.push_back(child.repkey());int s;
            try{
                // Logarithmic late move reductions; killers are exempt, PV nodes reduce one ply less.
                int reduction=0;
                if(pruning&&depth>=3&&i>=3&&quiet&&!check&&!child.check()&&!(m==killers[ply][0])&&!(m==killers[ply][1])){
                    reduction=int(0.75+std::log(double(depth))*std::log(double(i))/2.25)-(pv?1:0);
                    reduction=std::clamp(reduction,0,depth-2);
                }
                if(i==0)s=-search(child,depth-1,-beta,-alpha,ply+1,true,extensions);
                else {s=-search(child,depth-1-reduction,-alpha-1,-alpha,ply+1,true,extensions);if(reduction&&s>alpha)s=-search(child,depth-1,-alpha-1,-alpha,ply+1,true,extensions);if(s>alpha&&s<beta)s=-search(child,depth-1,-beta,-alpha,ply+1,true,extensions);}
            }catch(...){path.pop_back();throw;}path.pop_back();
            if(s>best){best=s;bestMove=m;}alpha=std::max(alpha,s);
            if(alpha>=beta){if(quiet){
                    if(!(killers[ply][0]==m)){killers[ply][1]=killers[ply][0];killers[ply][0]=m;}
                    // Reward the cutoff move, penalise quiets tried before it; gravity keeps values bounded.
                    int bonus=std::min(1600,16*depth*depth);
                    auto updateHistory=[&](Move q,int delta){int &h=history[b.turn][q.from][q.to];h+=delta-h*abs(delta)/16384;};
                    updateHistory(m,bonus);for(int k=0;k<triedCount;k++)updateHistory(tried[k],-bonus);
                }break;}
            if(quiet&&triedCount<64)tried[triedCount++]=m;
        }
        slot={key,bestMove,normalize(best,ply),depth,best>=beta?1:best<=oldAlpha?2:0,b.half};if(ply==0)rootBest=bestMove;return best;
    }
    void run(Board b,int maxdepth=64){start=std::chrono::steady_clock::now();nodes=0;rootBest={};nullPly=-1;path.clear();path.push_back(b.repkey());net.init(b,acc[0]);std::memset(history,0,sizeof(history));for(auto &k:killers)k[0]=k[1]={};
        List legal=b.legal();Move completed=legal.n?legal.v[0]:Move{};int previous=0;
        for(int depth=1;depth<=maxdepth&&legal.n;depth++){int delta=depth>=4?30:INF,alpha=std::max(-INF,previous-delta),beta=std::min(INF,previous+delta),score=0;
            try{while(true){score=search(b,depth,alpha,beta,0);if(score>alpha&&score<beta)break;if(alpha==-INF&&beta==INF)break;delta*=2;alpha=std::max(-INF,score-delta);beta=std::min(INF,score+delta);}}catch(int){break;}
            completed=rootBest.valid()?rootBest:completed;previous=score;
            std::cout<<"info depth "<<depth<<" score ";if(abs(score)>MATE-MAXPLY)std::cout<<"mate "<<(score>0?1:-1)*((MATE-abs(score)+1)/2);else std::cout<<"cp "<<score;
            std::cout<<" nodes "<<nodes<<" nps "<<U64(nodes*1000/std::max(1.0,elapsed()))<<" time "<<int(elapsed())<<" pv "<<uci(completed)<<std::endl;
            // A new iteration usually costs more than all previous ones; don't start one we can't finish.
            if(abs(score)>MATE-MAXPLY||stop||elapsed()>=softMilliseconds)break;
        }std::cout<<"bestmove "<<uci(completed)<<std::endl;
    }
};
U64 perft(const Board&b,int depth){if(!depth)return 1;List list=b.legal();if(depth==1)return list.n;U64 n=0;for(int i=0;i<list.n;i++)n+=perft(b.moved(list.v[i]),depth-1);return n;}
int main(int argc,char**argv){for(auto &row:Z)for(auto &v:row)v=random64();for(auto &v:Zcastle)v=random64();for(auto &v:Zep)v=random64();Zside=random64();
    try{Network net;std::string model;for(int i=1;i<argc;i++)if(std::string(argv[i])=="--model"&&i+1<argc)model=argv[++i];if(!model.empty())net.load(model);
        Search engine(net);Board board;board.fen(START);engine.past={board.repkey()};std::thread worker;
        auto halt=[&](){if(worker.joinable()){engine.stop=true;worker.join();}};
        std::string line;
        while(std::getline(std::cin,line)){std::istringstream in(line);std::string cmd;in>>cmd;
            if(cmd=="uci"){std::cout<<"id name ScratchChess Native 0.2\nid author ScratchChess\noption name Hash type spin default 64 min 1 max 1024\noption name Pruning type check default true\nuciok"<<std::endl;}
            else if(cmd=="isready")std::cout<<"readyok"<<std::endl;
            else if(cmd=="stop")halt();
            else if(cmd=="quit"){halt();break;}
            else if(cmd=="ucinewgame"){halt();std::fill(engine.tt.begin(),engine.tt.end(),Entry{});}
            else if(cmd=="setoption"){halt();std::string n,name,v;in>>n>>name>>v;if(name=="Pruning"){std::string x;in>>x;engine.pruning=x=="true";}else if(name=="Hash"){int mb;in>>mb;size_t count=1,limit=std::clamp(mb,1,1024)*1024ULL*1024/sizeof(Entry);while(count*2<=limit)count*=2;engine.tt.assign(count,Entry{});}}
            else if(cmd=="position"){halt();std::string mode;in>>mode;if(mode=="startpos")board.fen(START);else {std::string fen,t;for(int i=0;i<6&&in>>t;i++)fen+=t+" ";board.fen(fen);}engine.past={board.repkey()};std::string token;in>>token;while(in>>token){List list=board.legal();bool found=false;for(int i=0;i<list.n;i++)if(uci(list.v[i])==token){board=board.moved(list.v[i]);engine.past.push_back(board.repkey());found=true;break;}if(!found)throw std::runtime_error("illegal position move "+token);}}
            else if(cmd=="speedtest"){halt();if(!net.w)throw std::runtime_error("network required");
                // Isolated timings: which part of a node dominates at this network width?
                List l=board.legal();Acc a,d;net.init(board,a);std::vector<Board> kids;for(int i=0;i<l.n;i++)kids.push_back(board.moved(l.v[i]));
                auto time=[&](auto fn){auto t=std::chrono::steady_clock::now();volatile float sink=0;int reps=0;while(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()<0.5){for(int i=0;i<l.n;i++)sink+=fn(i);reps+=l.n;}return 1e9*std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/reps;};
                std::cout<<"ns movegen+make "<<time([&](int i){return float(board.moved(l.v[i]).check());})<<"\n";
                std::cout<<"ns legal() per move "<<time([&](int){return float(board.legal().n);})/l.n<<"\n";
                std::cout<<"ns update "<<time([&](int i){net.update(board,kids[i],a,d,l.v[i]);return d.a[0][0];})<<"\n";
                std::cout<<"ns value "<<time([&](int i){return net.value(board,a);})<<std::endl;}
            else if(cmd=="perft"){halt();int depth;in>>depth;std::cout<<"nodes "<<perft(board,depth)<<std::endl;}
            else if(cmd=="legal"){halt();List l=board.legal();for(int i=0;i<l.n;i++)std::cout<<uci(l.v[i])<<" ";std::cout<<std::endl;}
            else if(cmd=="eval"){halt();if(!net.w)throw std::runtime_error("network required");Acc a;net.init(board,a);std::cout.precision(9);std::cout<<"value "<<net.value(board,a)<<std::endl;}
            else if(cmd=="verifyacc"){halt();if(!net.w)throw std::runtime_error("network required");Acc a;net.init(board,a);List l=board.legal();float error=0;for(int i=0;i<l.n;i++){Board c=board.moved(l.v[i]);Acc incremental,fresh;net.update(board,c,a,incremental,l.v[i]);net.init(c,fresh);for(int s=0;s<2;s++)for(int j=0;j<net.w;j++)error=std::max(error,std::abs(incremental.a[s][j]-fresh.a[s][j]));}std::cout<<"error "<<error<<std::endl;}
            else if(cmd=="go"){halt();if(!net.w)throw std::runtime_error("network required");int depth=64;double ms=1000;U64 nodes=~U64(0);std::string token;double wt=-1,bt=-1,wi=0,bi=0,mt=-1;int moves=30;bool explicitLimit=false;
                while(in>>token){if(token=="depth"){in>>depth;explicitLimit=true;}else if(token=="nodes"){in>>nodes;explicitLimit=true;}else if(token=="infinite")explicitLimit=true;else if(token=="movetime")in>>mt;else if(token=="wtime")in>>wt;else if(token=="btime")in>>bt;else if(token=="winc")in>>wi;else if(token=="binc")in>>bi;else if(token=="movestogo")in>>moves;}
                if(explicitLimit)ms=1e15;double soft=ms,clock=board.turn?bt:wt,inc=board.turn?bi:wi;
                if(clock>=0){soft=std::max(1.0,std::min(clock*0.05,clock/std::max(1,moves)+inc*0.7));ms=std::max(1.0,std::min(clock*0.3,soft*3));}
                if(mt>=0)soft=ms=std::max(1.0,mt*0.95);
                engine.stop=false;engine.milliseconds=ms;engine.softMilliseconds=soft;engine.nodeLimit=nodes;worker=std::thread([&,b=board,depth](){engine.run(b,depth);});
            }
        }halt();
    }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}return 0;
}
