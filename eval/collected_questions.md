# 공개 Q&A 게시판에서 모은 실제 질문

출처는 LS ELECTRIC Solution Square의 Q&A 게시판이다. 2026-09-22에 모았다.

**여기에는 글 제목과 주소만 적는다.** 질문 본문과 답변 본문은 저작권이 각 작성자와
LS ELECTRIC에 있으므로 저장하지 않는다. 평가셋에 넣을 때도 질문을 다시 쓰고
출처 주소만 `note` 칸에 남긴다.

## 어떻게 가져왔나

게시판은 자바스크립트로 본문을 그리는 구조라 단순 HTTP 가져오기로는 목록도 본문도
오지 않는다. 내부 API(`/api/guest/ssqData/document/community/filter`)를 찾아 직접
불러 봤으나 서버가 500을 냈다(`period`가 정수여야 하고, 그걸 맞춰도 다른 칸에서
`eq(null)`이 났다). 로그인 세션이 필요한 것으로 보인다.

그래서 검색엔진 색인에서 제목과 주소만 가져왔다. Q&A 글은 **제목이 곧 질문**이라
제목만으로도 질문의 성격은 알 수 있다.

## 모은 것

| 번호 | 제목 | 제품명 | 매뉴얼로 답할 수 있나 |
|---|---|---|---|
| [30811](https://ssq.ls-electric.com/kr/ko/community/qna/document/30811) | 인버터 장기보관시 유의사항 문의 | 없음 | 가능 |
| [15002](https://ssq.ls-electric.com/kr/ko/community/qna/document/15002) | 인버터 오버홀 주기와 내용연수 및 사용기간 산정 | 없음 | 일부 |
| [21105](https://ssq.ls-electric.com/kr/ko/community/qna/document/21105) | 인버터 정격 전류가 모터 정격 전류 두 배 이상 일 때 설정값 | 없음 | 가능 |
| [15533](https://ssq.ls-electric.com/kr/ko/community/qna/document/15533) | G100 인버터 Q1/EG 결선 방법 | **G100** | 가능 |
| [21057](https://ssq.ls-electric.com/kr/ko/community/qna/document/21057) | G100 인버터 고장 이력 보는 법 | **G100** | 가능 |
| [23814](https://ssq.ls-electric.com/kr/ko/community/qna/document/23814) | S100 인버터 저속에서 구동불가 관련 문의드립니다 | **S100** | 일부 |
| [25405](https://ssq.ls-electric.com/kr/ko/community/qna/document/25405) | 제동저항 설정 관련 | 없음 | 가능 |
| [18351](https://ssq.ls-electric.com/kr/ko/community/qna/document/18351) | 인버터 에러 Out | 없음 | 불명확 (질문이 너무 짧음) |
| [17404](https://ssq.ls-electric.com/kr/ko/community/qna/document/17404) | 인버터 다중모터제어, 동기제어 문의드립니다 | 없음 | 일부 |
| [23818](https://ssq.ls-electric.com/kr/ko/community/qna/document/23818) | 1대의 인버터로 여러 모터 각각 다른 주파수로 제어 | 없음 | 일부 |
| [16319](https://ssq.ls-electric.com/kr/ko/community/qna/document/16319) | 인버터 용량 관련 다시 질문드립니다 | 없음 | 불명확 |
| [25065](https://ssq.ls-electric.com/kr/ko/community/qna/document/25065) | 220V 단상 인버터 제품 선정 관련 질의 | 없음 | 아니오 (제품 선정) |
| [10442](https://ssq.ls-electric.com/kr/ko/community/qna/document/10442) | 인버터류 차단기 선정 문의입니다 | 없음 | 일부 |
| [26272](https://ssq.ls-electric.com/kr/ko/community/qna/document/26272) | 인버터 수리 AS 관련 | 없음 | 아니오 (서비스 문의) |
| [11858](https://ssq.ls-electric.com/kr/ko/community/qna/document/11858) | LS PLC 와 LS 인버터 결선 질문있습니다 | 없음 | 아니오 (PLC 쪽) |
| [25229](https://ssq.ls-electric.com/kr/ko/community/qna/document/25229) | XGI 래더입니다. 인버터 운전지령 관련된 것입니다 | 없음 | 아니오 (PLC 쪽) |

## 이걸 보고 알게 된 것

### 1. 실제 질문은 제품명을 거의 안 쓴다

16건 중 제품명이 있는 것은 **3건뿐**이다. G100 둘, S100 하나다.

지금 평가셋 30문항 중 26개가 제품명을 문장 맨 앞에 달고 있다. 전부 만들어 쓴
질문이라 그렇다. 비율이 **87%와 19%로 갈린다.**

제품명으로 검색 범위를 좁히는 수정(#52)은 실제 질문의 다섯 중 넷에서 **아무 일도
하지 않는다.** 검색 적중 79.2%라는 수치는 "제품명이 적힌 질문에서"라는 조건 없이
쓰면 안 된다.

이건 추측이 아니라 세어 본 것이다. 그리고 평가셋을 만들어 쓸 때 어떤 편향이
생기는지를 보여 주는 실물 사례다.

### 2. 매뉴얼로 못 답하는 질문이 상당수다

16건 중 4건이 제품 선정, 수리 서비스, PLC 연동처럼 매뉴얼 밖의 것이다. 지금
평가셋의 답변 불가 6문항은 가격, 납기, 타사 비교, 업체 소개, 자격증인데, **실제
게시판의 답변 불가는 성격이 다르다.** "인접 제품군 질문"이 큰 몫이다.

### 3. 질문이 짧고 맥락이 본문에 있다

"인버터 에러 Out" 같은 제목은 그 자체로는 무엇을 묻는지 모른다. 본문에 모델명과
증상이 적혀 있을 것이다. 본문을 못 가져왔으므로 이런 건은 평가셋에 넣을 수 없다.

## 다음에 할 일

1. 본문을 볼 수 있는 방법을 찾는다. 로그인이 필요하면 수행자가 직접 열어 질문
   문장을 옮겨 적는 편이 빠르다
2. 본문 없이도 쓸 수 있는 것부터 평가셋에 넣는다. 위 표에서 "가능"인 4건이다.
   정답 근거는 **반드시 원본 PDF에서 읽는다.** 시스템 출력에서 베끼지 않는다
3. 제품명 없는 질문이 늘면 검색 적중이 떨어질 것이다. 그 수치를 따로 낸다.
   지금 수치와 섞으면 둘 다 못 쓴다

## 개정 이력

| 날짜 | 고친 곳 | 원래 내용 | 바꾼 내용 | 왜 바꿨나 |
| --- | --- | --- | --- | --- |
| 2026-09-22 | 신규 작성 | 없음 | 문서 생성 | 평가 질문이 전부 생성이라는 한계를 메우려고 실제 질문을 모음. 모으는 과정에서 제품명 사용 비율이 87%와 19%로 갈린다는 것이 드러나, 그 사실을 남기지 않으면 검색 적중 79.2%를 조건 없이 인용하게 됨 |
